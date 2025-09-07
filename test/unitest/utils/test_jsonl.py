import asyncio
import tempfile
from pathlib import Path
from typing import Any

import pytest

from protoss.utils.jsonl import JSONL, AsyncJSONL, JSONDecodeError, json_dumps, json_loads


class TestUtilityFunctions:
    """测试 json_dumps 和 json_loads 工具函数"""

    def test_json_dumps_basic(self):
        """测试 json_dumps 基本功能"""
        data = {'name': 'test', 'value': 123}
        result = json_dumps(data)
        assert isinstance(result, str)
        assert '"name":"test"' in result
        assert '"value":123' in result

    def test_json_dumps_sort_keys(self):
        """测试 json_dumps 的 sort_keys 参数"""
        data = {'z': 1, 'a': 2, 'm': 3}
        result = json_dumps(data, sort_keys=True)
        # 应该按字母顺序排序
        assert result.index('"a"') < result.index('"m"') < result.index('"z"')

    def test_json_loads_valid_json(self):
        """测试 json_loads 加载有效 JSON"""
        json_str = '{"name": "test", "value": 123}'
        result = json_loads(json_str)
        assert result == {'name': 'test', 'value': 123}

    def test_json_loads_array(self):
        """测试 json_loads 加载 JSON 数组"""
        json_str = '[1, 2, 3, "test"]'
        result = json_loads(json_str)
        assert result == [1, 2, 3, 'test']

    def test_json_loads_fallback_to_stdlib(self):
        """测试 json_loads 在 orjson 失败时回退到标准库"""
        # 测试包含 NaN 的 JSON（orjson 不支持）
        json_str = '{"value": NaN}'
        result = json_loads(json_str)
        # Python's json.loads correctly parses NaN as float('nan')
        import math

        assert 'value' in result
        assert math.isnan(result['value'])

        json_str = (
            '{"emoji": "\\ud83d\\ude80", "chinese": "\\u6d4b\\u8bd5", "special": "\\ud83d\\n\\t"}'
        )
        result = json_loads(json_str)
        assert result == {'emoji': '🚀', 'chinese': '测试', 'special': '\ud83d\n\t'}

    def test_json_roundtrip(self):
        """测试 dumps 和 loads 的往返转换"""
        original = {'complex': {'nested': [1, 2, {'deep': 'value'}]}}
        dumped = json_dumps(original)
        loaded = json_loads(dumped)
        assert loaded == original

    def test_json_dumps_with_numpy_types(self):
        """测试 json_dumps 处理 numpy 类型"""
        import numpy as np

        data = {'int': np.int64(42), 'float': np.float64(3.14)}
        result = json_dumps(data)
        assert '"int":42' in result
        assert '"float":3.14' in result

    def test_json_loads_invalid_json_raises_error(self):
        """测试 json_loads 对无效 JSON 抛出 JSONDecodeError"""

        invalid_json_cases = [
            '{"invalid": json}',  # 未引号的值
            '{"missing": }',  # 缺少值
            '{"trailing":,}',  # 尾随逗号
            'invalid json',  # 完全无效的格式
            '',  # 空字符串
            '{',  # 不完整的对象
            '[',  # 不完整的数组
        ]

        for invalid_json in invalid_json_cases:
            with pytest.raises(JSONDecodeError):
                json_loads(invalid_json)

    def test_json_loads_unicode_handling(self):
        """测试 json_loads 处理 Unicode 字符"""
        unicode_json = '{"emoji": "🚀", "chinese": "测试", "special": "\\n\\t"}'
        result = json_loads(unicode_json)
        assert result['emoji'] == '🚀'
        assert result['chinese'] == '测试'
        assert result['special'] == '\n\t'

    def test_json_dumps_indent(self):
        """测试 json_dumps 的 indent 参数"""
        data = {'name': 'test', 'nested': {'key': 'value'}}

        # Test compact format (default)
        compact = json_dumps(data)
        assert '\n' not in compact

        # Test with indent=2
        pretty = json_dumps(data, indent=2)
        assert '\n' in pretty
        assert '  ' in pretty  # Should have indentation

        # Test with indent=0 (should be same as compact)
        indent0 = json_dumps(data, indent=0)
        assert indent0 == compact

        # Test with indent=2 and sort_keys
        pretty_sorted = json_dumps(data, indent=2, sort_keys=True)
        assert '\n' in pretty_sorted
        assert '  ' in pretty_sorted

    def test_json_dumps_ensure_ascii(self):
        """测试 json_dumps 的 ensure_ascii 参数"""
        data = {'emoji': '🚀', 'chinese': '测试', 'special': '\ud83d\n\t'}
        result = json_dumps(data, ensure_ascii=True)
        assert (
            result
            == '{"emoji": "\\ud83d\\ude80", "chinese": "\\u6d4b\\u8bd5", "special": "\\ud83d\\n\\t"}'
        )


class TestJSONL:
    """测试同步版本的 JSONL 类"""

    def test_write_and_read(self):
        """测试基本的写入和读取功能"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name

        try:
            # 写入数据
            with JSONL(temp_path).open('w') as jsonl:
                jsonl.write({'id': 1, 'name': 'Alice', 'age': 25})
                jsonl.write({'id': 2, 'name': 'Bob', 'age': 30})
                jsonl.write({'id': 3, 'name': 'Charlie', 'age': 35})

            # 读取数据
            with JSONL(temp_path).open('r') as jsonl:
                items = list(jsonl)
                assert len(items) == 3
                assert items[0] == {'id': 1, 'name': 'Alice', 'age': 25}
                assert items[1] == {'id': 2, 'name': 'Bob', 'age': 30}
                assert items[2] == {'id': 3, 'name': 'Charlie', 'age': 35}

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_random_access(self):
        """测试随机访问功能"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name

        try:
            # 写入数据
            with JSONL(temp_path).open('w') as jsonl:
                jsonl.write({'id': 1, 'name': 'Alice'})
                jsonl.write({'id': 2, 'name': 'Bob'})
                jsonl.write({'id': 3, 'name': 'Charlie'})

            # 测试随机访问
            with JSONL(temp_path).open('r') as jsonl:
                assert jsonl[0] == {'id': 1, 'name': 'Alice'}
                assert jsonl[1] == {'id': 2, 'name': 'Bob'}
                assert jsonl[2] == {'id': 3, 'name': 'Charlie'}
                assert jsonl[-1] == {'id': 3, 'name': 'Charlie'}
                assert jsonl[-2] == {'id': 2, 'name': 'Bob'}

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_empty_file(self):
        """测试空文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name

        try:
            with JSONL(temp_path).open('r') as jsonl:
                items = list(jsonl)
                assert len(items) == 0
                assert len(jsonl) == 0

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_append_mode(self):
        """测试追加模式"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name

        try:
            # 第一次写入
            with JSONL(temp_path).open('w') as jsonl:
                jsonl.write({'id': 1, 'name': 'Alice'})

            # 追加写入
            with JSONL(temp_path).open('a') as jsonl:
                jsonl.write({'id': 2, 'name': 'Bob'})

            # 读取所有数据
            with JSONL(temp_path).open('r') as jsonl:
                items = list(jsonl)
                assert len(items) == 2
                assert items[0] == {'id': 1, 'name': 'Alice'}
                assert items[1] == {'id': 2, 'name': 'Bob'}

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_index_cache(self):
        """测试索引缓存功能"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name

        try:
            # 写入数据
            with JSONL(temp_path).open('w') as jsonl:
                jsonl.write({'id': 1, 'name': 'Alice'})
                jsonl.write({'id': 2, 'name': 'Bob'})

            # 第一次读取（会构建索引）
            with JSONL(temp_path).open('r') as jsonl:
                items = list(jsonl)
                assert len(items) == 2

            # 检查索引文件是否存在
            index_path = Path(temp_path).with_suffix('.jslindex')
            assert index_path.exists()

            # 第二次读取（使用缓存的索引）
            with JSONL(temp_path).open('r') as jsonl:
                items = list(jsonl)
                assert len(items) == 2

        finally:
            Path(temp_path).unlink(missing_ok=True)
            Path(temp_path).with_suffix('.jslindex').unlink(missing_ok=True)


class TestAsyncJSONL:
    """测试异步版本的 AsyncJSONL 类"""

    @pytest.mark.asyncio
    async def test_write_and_read(self):
        """测试基本的异步写入和读取功能"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name

        try:
            # 写入数据
            async with AsyncJSONL(temp_path).open('w') as jsonl:
                await jsonl.write({'id': 1, 'name': 'Alice', 'age': 25})
                await jsonl.write({'id': 2, 'name': 'Bob', 'age': 30})
                await jsonl.write({'id': 3, 'name': 'Charlie', 'age': 35})

            # 读取数据
            async with AsyncJSONL(temp_path).open('r') as jsonl:
                items = [item async for item in jsonl]
                assert len(items) == 3
                assert items[0] == {'id': 1, 'name': 'Alice', 'age': 25}
                assert items[1] == {'id': 2, 'name': 'Bob', 'age': 30}
                assert items[2] == {'id': 3, 'name': 'Charlie', 'age': 35}

        finally:
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_random_access(self):
        """测试异步随机访问功能"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name

        try:
            # 写入数据
            async with AsyncJSONL(temp_path).open('w') as jsonl:
                await jsonl.write({'id': 1, 'name': 'Alice'})
                await jsonl.write({'id': 2, 'name': 'Bob'})
                await jsonl.write({'id': 3, 'name': 'Charlie'})

            # 测试随机访问
            async with AsyncJSONL(temp_path).open('r') as jsonl:
                assert await jsonl[0] == {'id': 1, 'name': 'Alice'}
                assert await jsonl[1] == {'id': 2, 'name': 'Bob'}
                assert await jsonl[2] == {'id': 3, 'name': 'Charlie'}
                assert await jsonl[-1] == {'id': 3, 'name': 'Charlie'}
                assert await jsonl[-2] == {'id': 2, 'name': 'Bob'}

        finally:
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_empty_file(self):
        """测试空文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name

        try:
            async with AsyncJSONL(temp_path).open('r') as jsonl:
                items = [item async for item in jsonl]
                assert len(items) == 0
                assert len(jsonl) == 0

        finally:
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_append_mode(self):
        """测试异步追加模式"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name

        try:
            # 第一次写入
            async with AsyncJSONL(temp_path).open('w') as jsonl:
                await jsonl.write({'id': 1, 'name': 'Alice'})

            # 追加写入
            async with AsyncJSONL(temp_path).open('a') as jsonl:
                await jsonl.write({'id': 2, 'name': 'Bob'})

            # 读取所有数据
            async with AsyncJSONL(temp_path).open('r') as jsonl:
                items = [item async for item in jsonl]
                assert len(items) == 2
                assert items[0] == {'id': 1, 'name': 'Alice'}
                assert items[1] == {'id': 2, 'name': 'Bob'}

        finally:
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_index_cache(self):
        """测试异步索引缓存功能"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name

        try:
            # 写入数据
            async with AsyncJSONL(temp_path).open('w') as jsonl:
                await jsonl.write({'id': 1, 'name': 'Alice'})
                await jsonl.write({'id': 2, 'name': 'Bob'})

            # 第一次读取（会构建索引）
            async with AsyncJSONL(temp_path).open('r') as jsonl:
                items = [item async for item in jsonl]
                assert len(items) == 2

            # 检查索引文件是否存在
            index_path = Path(temp_path).with_suffix('.jslindex')
            assert index_path.exists()

            # 第二次读取（使用缓存的索引）
            async with AsyncJSONL(temp_path).open('r') as jsonl:
                items = [item async for item in jsonl]
                assert len(items) == 2

        finally:
            Path(temp_path).unlink(missing_ok=True)
            Path(temp_path).with_suffix('.jslindex').unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """测试并发访问"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name

        try:
            # 写入数据
            async with AsyncJSONL(temp_path).open('w') as jsonl:
                for i in range(10):
                    await jsonl.write({'id': i, 'name': f'User{i}'})

            # 并发读取
            async def read_items() -> list[dict[str, Any]]:
                async with AsyncJSONL(temp_path).open('r') as jsonl:
                    return [item async for item in jsonl]

            # 创建多个并发任务
            tasks = [read_items() for _ in range(3)]
            results = await asyncio.gather(*tasks)

            # 验证所有结果都正确
            for result in results:
                assert len(result) == 10
                for i, item in enumerate(result):
                    assert item == {'id': i, 'name': f'User{i}'}

        finally:
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """测试错误处理"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name

        try:
            # 测试访问未打开的文件
            jsonl = AsyncJSONL(temp_path)
            with pytest.raises(ValueError, match='File not opened'):
                await jsonl[0]

            # 测试访问不存在的索引
            async with AsyncJSONL(temp_path).open('w') as jsonl:
                await jsonl.write({'id': 1, 'name': 'Alice'})

            async with AsyncJSONL(temp_path).open('r') as jsonl:
                with pytest.raises(IndexError):
                    await jsonl[1]  # 只有一行数据，索引1不存在

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestJSONLComparison:
    """测试同步和异步版本的兼容性"""

    def test_sync_vs_async_same_result(self):
        """测试同步和异步版本产生相同的结果"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name

        try:
            test_data = [
                {'id': 1, 'name': 'Alice', 'age': 25},
                {'id': 2, 'name': 'Bob', 'age': 30},
                {'id': 3, 'name': 'Charlie', 'age': 35},
            ]

            # 使用同步版本写入
            with JSONL(temp_path).open('w') as jsonl:
                for item in test_data:
                    jsonl.write(item)

            # 使用同步版本读取
            with JSONL(temp_path).open('r') as jsonl:
                sync_items = list(jsonl)

            # 使用异步版本读取相同文件
            async def read_async() -> list[dict[str, Any]]:
                async with AsyncJSONL(temp_path).open('r') as jsonl:
                    return [item async for item in jsonl]

            async_items = asyncio.run(read_async())

            # 验证结果相同
            assert sync_items == async_items
            assert sync_items == test_data

        finally:
            Path(temp_path).unlink(missing_ok=True)


if __name__ == '__main__':
    pytest.main([__file__])
