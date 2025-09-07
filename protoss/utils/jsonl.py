import json
import os
import random
import tempfile
from collections.abc import AsyncGenerator, AsyncIterator, Generator
from contextlib import asynccontextmanager, contextmanager
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Self

import aiofiles
import numpy as np
import orjson
from loguru import logger


# Unified JSON decode error type
JSONDecodeError = (orjson.JSONDecodeError, json.JSONDecodeError)


def json_dumps(
    obj: Any, *, sort_keys: bool = False, indent: int | None = None, ensure_ascii: bool = False
) -> str:
    """
    Dump JSON data to a string using orjson with optimized options.

    Args:
        obj: Python object to serialize
        sort_keys: Whether to sort keys in the output (like standard json.dumps)
        indent: Number of spaces to indent for pretty-printing. None for compact format.
        ensure_ascii: Whether to ensure ASCII encoding.

    Returns:
        JSON string representation
    """
    # 如果开启了 ensure_ascii，直接使用 json 库的实现
    if ensure_ascii:
        return json.dumps(obj, ensure_ascii=True, indent=indent, sort_keys=sort_keys)

    option = orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY
    if sort_keys:
        option |= orjson.OPT_SORT_KEYS
    if indent is not None and indent > 0:
        option |= orjson.OPT_INDENT_2
    return orjson.dumps(obj, option=option, default=str).decode()


def json_loads(data: str) -> dict[str, Any]:
    """
    Load JSON data from a string.

    Args:
        data: The JSON string to load.

    Returns:
        The loaded JSON data.
    """
    try:
        return orjson.loads(data)
    except orjson.JSONDecodeError:
        # Fallback to standard library for edge cases like NaN
        return json.loads(data)


class BaseJSONL:
    """Base class for JSONL file operations."""

    def __init__(self, path: str | Path, strict_integrity: bool = False) -> None:
        self._path = Path(path)
        self._file = None
        self._line_index = None
        self._index_cache_dir = Path('/tmp/jsonl_index_cache/')
        self._index_cache_dir.mkdir(parents=True, exist_ok=True)
        self._strict_integrity = strict_integrity

    @property
    def line_index(self) -> list[int]:
        """Get line index."""
        return self._line_index if self._line_index is not None else []

    @property
    def total(self) -> int:
        """Get total number of lines."""
        if not Path(self._path).exists() or self.line_index is None:
            return 0
        return len(self.line_index)

    def __len__(self) -> int:
        """Get length."""
        return self.total

    @cached_property
    def index_file(self) -> Path:
        """Get index file path."""
        ifilename = f'{self._path.stem}.jslindex'
        adjacent_index_file = self._path.with_name(ifilename)
        if adjacent_index_file.exists() or os.access(self._path, os.W_OK):
            # 1. already has index file  2. writable file system
            return adjacent_index_file
        elif self._index_cache_dir:
            # otherwise, create index file in index cache dir
            p0 = Path(self._path.resolve())
            rel_path = p0.relative_to('/')
            ifile = self._index_cache_dir / str(rel_path)
            ifile.parent.mkdir(parents=True, exist_ok=True)
            ifile = ifile.parent / ifilename
            return ifile
        else:
            raise RuntimeError(f'Cannot create index file for {self._path}')

    def _atomic_create_index_file(self, line_index: list[int]) -> None:
        """
        Create index file atomically.

        Args:
            line_index: The begin position of each line, including the last line-end.

        Raises:
            OSError: If the index file cannot be created.
        """
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='wb', delete=False, dir=self.index_file.parent
            ) as tmp_file:
                np.array(line_index, dtype=np.int64).tofile(tmp_file)
                tmp_path = tmp_file.name
            # 设置正确的权限, 默认是 0o600
            os.chmod(tmp_path, 0o644)
            os.replace(tmp_path, self.index_file)
        except OSError:
            if tmp_path is not None and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise


class JSONL(BaseJSONL):
    """
    A simple JSONL file reader/writer

    Args:
        path: The path to the JSONL file.
        strict_integrity: Whether to check the integrity of the JSONL file strictly. If True, will sample 100 lines to check the integrity. If False, will only check file size.

    Example:
    ```
    with JSONL("data.jsonl").open('w') as f:
        f.write({"a": 1, "b": 2})
        f.write({"a": 3, "b": 4})

    with JSONL("data.jsonl").open('r') as f:
        for obj in f:
            print(obj)

        # seek to a specific line
        print(f[15])
    ```
    """

    _index_cache_dir: Path = Path('/tmp/jsonl_index_cache/')

    def __init__(self, path: str | Path, strict_integrity: bool = False) -> None:
        super().__init__(path, strict_integrity)

    @classmethod
    def set_index_cache_dir(cls, index_cache_dir: Path) -> None:
        cls._index_cache_dir = Path(index_cache_dir)
        cls._index_cache_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def open(
        self, mode: Literal['r', 'w', 'a'] = 'r', build_index: bool = True
    ) -> Generator[Self, None, None]:
        try:
            self._file = open(self._path, mode)  # noqa: SIM115
            if mode == 'r' and build_index and not self._load_index_file():
                self._build_line_index()
            yield self
        finally:
            self.close()

    def close(self) -> None:
        if self._file:
            self._file.close()

        self._file = None
        self._line_index = []

    def _build_line_index(self) -> None:
        assert self._file
        logger.info(f'building line index for {self._path}')
        line_index = [0]
        # add begin position of each line, including the last line-end
        for line in self._file:
            line_index.append(line_index[-1] + len(line.encode('utf-8')))
        self._atomic_create_index_file(line_index)
        self._line_index = line_index[:-1]  # remove the last line-end
        self._file.seek(0)

    def _load_index_file(self) -> bool:
        assert self._file
        _index_file = self.index_file
        if not _index_file.exists():
            return False

        # sanity check
        try:
            line_index = np.fromfile(_index_file, dtype=np.int64).tolist()
            assert len(line_index) > 0
            file_size = self._path.stat().st_size
            assert file_size == line_index[-1], (
                f'file size mismatch: {file_size} vs {line_index[-1]}'
            )
            if self._strict_integrity:
                for _ in range(min(100, len(line_index) - 1)):
                    i = random.randint(0, len(line_index) - 2)
                    self._file.seek(line_index[i])
                    data = self._file.readline()
                    json_loads(data)  # test data integrity
                    assert line_index[i + 1] == self._file.tell(), (
                        f'round {_} failed: index[{i}]={line_index[i]}'
                    )
        except (OSError, ValueError, AssertionError) as e:
            logger.warning(f'Error: {e}')
            logger.warning(f'Index file for `{self._path}` seems mismatched, rebuilding...')
            return False

        self._line_index = line_index[:-1]  # remove the last line-end
        self._file.seek(0)
        return True

    def write(self, data: dict[str, Any]) -> None:
        if self._file is None:
            raise ValueError('File not opened')
        self._file.write(json_dumps(data) + '\n')
        self._file.flush()

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._file is None:
            raise ValueError('File not opened')
        if self._line_index is None:
            raise ValueError('Line index not built')

        if index < 0:
            index += self.total
        if index < 0 or index >= self.total:
            raise IndexError

        self._file.seek(self._line_index[index])
        try:
            return json_loads(self._file.readline().strip())
        except Exception:
            logger.exception(f'Failed to load {self._path}:{index}')
            raise

    def __iter__(self) -> Self:
        if self._file:
            self._file.seek(0)

        return self

    def __next__(self) -> dict[str, Any]:
        if not self._file:
            raise StopIteration

        line = self._file.readline()
        if not line:
            raise StopIteration

        return json_loads(line.strip())


class AsyncJSONL(BaseJSONL):
    """
    An async version of JSONL file reader/writer

    Args:
        path: The path to the JSONL file.
        strict_integrity: Whether to check the integrity of the JSONL file strictly. If True, will sample 100 lines to check the integrity. If False, will only check file size.

    Example:
    ```
    # 推荐用法：与同步版本保持一致的语法
    async with AsyncJSONL("data.jsonl").open('w') as f:
        await f.write({"a": 1, "b": 2})
        await f.write({"a": 3, "b": 4})

    async with AsyncJSONL("data.jsonl").open('r') as f:
        async for obj in f:
            print(obj)

        # seek to a specific line
        print(await f[15])
    ```
    """

    def __init__(self, path: str | Path, strict_integrity: bool = False) -> None:
        super().__init__(path, strict_integrity)

    @asynccontextmanager
    async def open(
        self, mode: Literal['r', 'w', 'a'] = 'r', build_index: bool = True
    ) -> AsyncGenerator[Self, None]:
        try:
            self._file = await aiofiles.open(self._path, mode)
            if mode == 'r' and build_index and not await self._load_index_file():
                await self._build_line_index()
            yield self
        finally:
            await self.close()

    async def close(self) -> None:
        if self._file:
            await self._file.close()
            self._file = None
            self._line_index = []

    async def _build_line_index(self) -> None:
        assert self._file
        logger.info(f'building line index for {self._path}')
        line_index = [0]
        # add begin position of each line, including the last line-end
        async for line in self._file:
            line_index.append(line_index[-1] + len(line.encode('utf-8')))
        self._atomic_create_index_file(line_index)
        self._line_index = line_index[:-1]  # remove the last line-end
        await self._file.seek(0)

    async def _load_index_file(self) -> bool:
        assert self._file
        _index_file = self.index_file
        if not _index_file.exists():
            return False

        # sanity check
        try:
            line_index = np.fromfile(_index_file, dtype=np.int64).tolist()
            assert len(line_index) > 0
            file_size = self._path.stat().st_size
            assert file_size == line_index[-1], (
                f'file size mismatch: {file_size} vs {line_index[-1]}'
            )
            if self._strict_integrity:
                for _ in range(min(100, len(line_index) - 1)):
                    i = random.randint(0, len(line_index) - 2)
                    await self._file.seek(line_index[i])
                    data = await self._file.readline()
                    json_loads(data)  # test data integrity
                    assert line_index[i + 1] == await self._file.tell(), (
                        f'round {_} failed: index[{i}]={line_index[i]}'
                    )
        except (OSError, ValueError, AssertionError) as e:
            logger.warning(f'Error: {e}')
            logger.warning(f'Index file for `{self._path}` seems mismatched, rebuilding...')
            return False

        self._line_index = line_index[:-1]  # remove the last line-end
        await self._file.seek(0)
        return True

    async def write(self, data: dict[str, Any]) -> None:
        if self._file is None:
            raise ValueError('File not opened')
        await self._file.write(json_dumps(data) + '\n')
        await self._file.flush()

    async def __getitem__(self, index: int) -> dict[str, Any]:
        if self._file is None:
            raise ValueError('File not opened')
        if self._line_index is None:
            raise ValueError('Line index not built')

        if index < 0:
            index += self.total
        if index < 0 or index >= self.total:
            raise IndexError

        await self._file.seek(self._line_index[index])
        try:
            line = await self._file.readline()
            return json_loads(line.strip())
        except Exception:
            logger.exception(f'Failed to load {self._path}:{index}')
            raise

    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        if self._file:
            await self._file.seek(0)

        while True:
            if not self._file:
                break

            line = await self._file.readline()
            if not line:
                break

            yield json_loads(line.strip())
