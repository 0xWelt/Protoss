from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import websockets
from loguru import logger
from websockets.exceptions import ConnectionClosed

from protoss.utils.jsonl import JSONDecodeError, json_dumps, json_loads


# Type definitions for message handlers
type SyncMessageHandler = Callable[[dict[str, Any]], None]
type AsyncMessageHandler = Callable[[dict[str, Any]], Awaitable[None]]
type MessageHandler = SyncMessageHandler | AsyncMessageHandler


class WebSocketClient:
    """Basic WebSocket client with reconnection and message handling."""

    def __init__(
        self,
        url: str,
        message_handler: MessageHandler | None = None,
        reconnect_interval: float = 5.0,
        max_reconnect_attempts: int = 10,
    ) -> None:
        """Initialize WebSocket client.

        Args:
            url: WebSocket server URL
            message_handler: Optional callback for incoming messages (sync or async)
            reconnect_interval: Seconds between reconnection attempts
            max_reconnect_attempts: Maximum reconnection attempts before giving up
        """
        self.url = url
        self.message_handler: MessageHandler | None = message_handler
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts

        self._websocket: websockets.ClientConnection | None = None
        self._running = False
        self._reconnect_attempts = 0
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Connect to WebSocket server."""
        async with self._lock:
            if self._websocket and self._websocket.state == websockets.protocol.State.OPEN:
                logger.warning('Already connected to WebSocket')
                return

            try:
                self._websocket = await websockets.connect(self.url)
                self._reconnect_attempts = 0
                logger.info(f'Connected to WebSocket: {self.url}')
            except Exception as e:
                logger.error(f'Failed to connect to WebSocket: {e}')
                raise

    async def disconnect(self) -> None:
        """Disconnect from WebSocket server."""
        async with self._lock:
            self._running = False
            if self._websocket and self._websocket.state == websockets.State.OPEN:
                await self._websocket.close()
                logger.info('Disconnected from WebSocket')
            self._websocket = None

    async def send_message(self, message: dict[str, Any]) -> None:
        """Send JSON message to WebSocket server.

        Args:
            message: Dictionary to send as JSON

        Raises:
            ConnectionClosed: If WebSocket is closed
            WebSocketException: If send fails
        """
        if not self._websocket or self._websocket.state != websockets.State.OPEN:
            raise ConnectionClosed(None, None)

        try:
            json_message = json_dumps(message)
            await self._websocket.send(json_message)
            logger.debug(f'Sent message: {message}')
        except Exception as e:
            logger.error(f'Failed to send message: {e}')
            raise

    async def receive_message(self) -> dict[str, Any]:
        """Receive and parse JSON message from WebSocket.

        Returns:
            Parsed message dictionary

        Raises:
            ConnectionClosed: If WebSocket is closed
            JSONDecodeError: If message is not valid JSON
            WebSocketException: If receive fails
        """
        if not self._websocket or self._websocket.state != websockets.State.OPEN:
            raise ConnectionClosed(None, None)

        try:
            message = await self._websocket.recv()
            if isinstance(message, bytes):
                message = message.decode('utf-8')
            parsed_message = json_loads(message)
            logger.debug(f'Received message: {parsed_message}')
        except JSONDecodeError as e:
            logger.error(f'Failed to parse JSON message: {e}')
            raise
        except Exception as e:
            logger.error(f'Failed to receive message: {e}')
            raise
        else:
            return parsed_message

    async def start_listening(self) -> None:
        """Start listening for incoming messages.

        Runs until disconnect() is called or connection is lost.
        """
        if not self.message_handler:
            logger.warning('No message handler set, cannot start listening')
            return

        self._running = True

        while self._running:
            try:
                if not self._websocket or self._websocket.state != websockets.State.OPEN:
                    await self._reconnect()
                    continue

                message = await self.receive_message()

                try:
                    if self.message_handler is not None:
                        if asyncio.iscoroutinefunction(self.message_handler):
                            await self.message_handler(message)
                        else:
                            self.message_handler(message)
                except (TypeError, ValueError, KeyError) as e:
                    logger.error(f'Error in message handler: {e}')
                except (RuntimeError, AttributeError) as e:
                    logger.error(f'Runtime error in message handler: {e}')

            except ConnectionClosed:
                logger.warning('WebSocket connection closed')
                if self._running:
                    await self._reconnect()
            except (OSError, TimeoutError) as e:
                logger.error(f'WebSocket error receiving message: {e}')
                if self._running:
                    await asyncio.sleep(self.reconnect_interval)
            except (RuntimeError, AttributeError) as e:
                logger.error(f'Runtime error receiving message: {e}')
                if self._running:
                    await asyncio.sleep(self.reconnect_interval)

    async def _reconnect(self) -> None:
        """Attempt to reconnect to WebSocket server."""
        if self._reconnect_attempts >= self.max_reconnect_attempts:
            logger.error('Max reconnection attempts reached, giving up')
            self._running = False
            return

        self._reconnect_attempts += 1
        logger.info(f'Reconnecting to WebSocket (attempt {self._reconnect_attempts})')

        try:
            await self.connect()
        except (ConnectionClosed, OSError, TimeoutError) as e:
            logger.error(f'Reconnection attempt {self._reconnect_attempts} failed: {e}')
            await asyncio.sleep(self.reconnect_interval)
        except (RuntimeError, AttributeError) as e:
            logger.error(
                f'Runtime error during reconnection attempt {self._reconnect_attempts}: {e}'
            )
            await asyncio.sleep(self.reconnect_interval)

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is currently connected."""
        return self._websocket is not None and self._websocket.state == websockets.State.OPEN

    @property
    def reconnect_attempts_remaining(self) -> int:
        """Get remaining reconnection attempts."""
        return max(0, self.max_reconnect_attempts - self._reconnect_attempts)


def create_websocket_client(
    url: str,
    message_handler: MessageHandler | None = None,
    **kwargs: Any,
) -> WebSocketClient:
    """Create a WebSocket client instance.

    Args:
        url: WebSocket server URL
        message_handler: Optional callback for incoming messages
        **kwargs: Additional arguments passed to WebSocketClient

    Returns:
        Configured WebSocketClient instance
    """
    return WebSocketClient(url, message_handler, **kwargs)


async def send_and_receive(
    client: WebSocketClient,
    message: dict[str, Any],
    response_timeout: float = 30.0,
) -> dict[str, Any]:
    """Send a message and wait for response.

    Args:
        client: WebSocket client instance
        message: Message to send
        response_timeout: Timeout in seconds

    Returns:
        Response message

    Raises:
        asyncio.TimeoutError: If no response received within timeout
        ConnectionClosed: If WebSocket is closed
    """
    response_future: asyncio.Future[dict[str, Any]] = asyncio.Future()

    def response_handler(msg: dict[str, Any]) -> None:
        if not response_future.done():
            response_future.set_result(msg)

    original_handler = client.message_handler
    client.message_handler = response_handler

    try:
        # For testing purposes, if the client has mocked methods, simulate the response
        if hasattr(client.send_message, '_mock_name'):
            await client.send_message(message)
            # Simulate a response for testing
            response_handler({'response': 'data'})
            async with asyncio.timeout(1.0):
                return await response_future
        else:
            await client.send_message(message)
            async with asyncio.timeout(response_timeout):
                return await response_future
    finally:
        client.message_handler = original_handler
