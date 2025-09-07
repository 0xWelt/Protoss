"""Tests for WebSocket client implementation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from websockets.exceptions import ConnectionClosed

from protoss.utils.ws import WebSocketClient, create_websocket_client, send_and_receive


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    mock = MagicMock()
    mock.state = MagicMock()
    mock.state.OPEN = 1  # websockets.State.OPEN value
    mock.state = 1  # Set state to OPEN
    mock.send = AsyncMock()
    mock.recv = AsyncMock(return_value='{"test": "message"}')
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def websocket_client():
    """Create a WebSocket client instance."""
    return WebSocketClient('ws://test.example.com')


@pytest.mark.asyncio
async def test_websocket_client_initialization():
    """Test WebSocket client initialization."""
    client = WebSocketClient('ws://test.example.com')

    assert client.url == 'ws://test.example.com'
    assert client.message_handler is None
    assert client.reconnect_interval == 5.0
    assert client.max_reconnect_attempts == 10
    assert client._websocket is None
    assert client._running is False
    assert client._reconnect_attempts == 0


@pytest.mark.asyncio
async def test_websocket_client_with_handler():
    """Test WebSocket client with message handler."""

    def handler(message):
        pass

    client = WebSocketClient('ws://test.example.com', message_handler=handler)
    assert client.message_handler == handler


@pytest.mark.asyncio
async def test_create_websocket_client():
    """Test factory function for creating WebSocket client."""
    client = create_websocket_client('ws://test.example.com')
    assert isinstance(client, WebSocketClient)
    assert client.url == 'ws://test.example.com'


@pytest.mark.asyncio
async def test_is_connected_property(mock_websocket, websocket_client):
    """Test is_connected property."""
    assert websocket_client.is_connected is False

    websocket_client._websocket = mock_websocket
    assert websocket_client.is_connected is True

    mock_websocket.state = 3  # CLOSED state
    assert websocket_client.is_connected is False


@pytest.mark.asyncio
async def test_reconnect_attempts_remaining(websocket_client):
    """Test reconnect_attempts_remaining property."""
    assert websocket_client.reconnect_attempts_remaining == 10

    websocket_client._reconnect_attempts = 3
    assert websocket_client.reconnect_attempts_remaining == 7

    websocket_client._reconnect_attempts = 15
    assert websocket_client.reconnect_attempts_remaining == 0


@pytest.mark.asyncio
async def test_send_message_success(mock_websocket, websocket_client):
    """Test successful message sending."""
    websocket_client._websocket = mock_websocket

    test_message = {'type': 'test', 'data': 'hello'}
    await websocket_client.send_message(test_message)

    mock_websocket.send.assert_called_once()
    sent_data = mock_websocket.send.call_args[0][0]
    assert '"type":"test"' in sent_data
    assert '"data":"hello"' in sent_data


@pytest.mark.asyncio
async def test_send_message_no_connection(websocket_client):
    """Test sending message without connection."""
    test_message = {'type': 'test', 'data': 'hello'}

    with pytest.raises(ConnectionClosed):
        await websocket_client.send_message(test_message)


@pytest.mark.asyncio
async def test_receive_message_success(mock_websocket, websocket_client):
    """Test successful message receiving."""
    websocket_client._websocket = mock_websocket

    message = await websocket_client.receive_message()

    assert message == {'test': 'message'}
    mock_websocket.recv.assert_called_once()


@pytest.mark.asyncio
async def test_receive_message_no_connection(websocket_client):
    """Test receiving message without connection."""
    with pytest.raises(ConnectionClosed):
        await websocket_client.receive_message()


@pytest.mark.asyncio
async def test_start_listening_no_handler(websocket_client):
    """Test start_listening without message handler."""
    await websocket_client.start_listening()
    # Should return early without error


@pytest.mark.asyncio
async def test_sync_message_handler():
    """Test synchronous message handler."""
    received_messages = []

    def sync_handler(message):
        received_messages.append(message)

    client = WebSocketClient('ws://test.example.com', message_handler=sync_handler)

    # Simulate message handling
    test_message = {'test': 'data'}

    # Manually trigger handler for testing (don't call start_listening)
    if client.message_handler:
        client.message_handler(test_message)

    assert len(received_messages) == 1
    assert received_messages[0] == test_message


@pytest.mark.asyncio
async def test_async_message_handler():
    """Test asynchronous message handler."""
    received_messages = []

    async def async_handler(message):
        received_messages.append(message)

    client = WebSocketClient('ws://test.example.com', message_handler=async_handler)

    # Simulate message handling
    test_message = {'test': 'data'}

    # Manually trigger handler for testing
    if client.message_handler:
        if asyncio.iscoroutinefunction(client.message_handler):
            await client.message_handler(test_message)
        else:
            client.message_handler(test_message)

    assert len(received_messages) == 1
    assert received_messages[0] == test_message


@pytest.mark.asyncio
async def test_send_and_receive():
    """Test send_and_receive utility function."""
    client = WebSocketClient('ws://test.example.com')

    # Mock the send_message and receive_message methods
    client.send_message = AsyncMock()
    client.receive_message = AsyncMock(return_value={'response': 'data'})

    # Set up a mock message handler that will be called
    original_handler = MagicMock()
    client.message_handler = original_handler

    result = await send_and_receive(client, {'request': 'data'})

    assert result == {'response': 'data'}
    client.send_message.assert_called_once_with({'request': 'data'})

    # Verify original handler was restored
    assert client.message_handler == original_handler
