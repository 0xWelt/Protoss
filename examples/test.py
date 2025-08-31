import asyncio
import os
from typing import Any

from loguru import logger
from okx.websocket.WsPrivateAsync import WsPrivateAsync

from protoss.utils.jsonl import json_loads


def load_config() -> dict[str, Any]:
    """Load configuration from config.json file"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, encoding='utf-8') as file:
        config = json_loads(file.read())
    return config


def callbackFunc(message: str):
    logger.info(message)


async def main():
    # Load configuration from config.json
    config = load_config()
    if not config:
        logger.error('Failed to load configuration. Exiting.')
        return

    # Extract configuration values
    api_key = config.get('apiKey')
    secret_key = config.get('secretKey')
    passphrase = config.get('passphrase')

    if not all([api_key, secret_key, passphrase]):
        logger.error('Error: Missing required configuration values (apiKey, secretKey, passphrase)')
        return

    ws = WsPrivateAsync(
        apiKey=api_key,
        passphrase=passphrase,
        secretKey=secret_key,
        url='wss://ws.okx.com:8443/ws/v5/private',
        useServerTime=False,
    )
    await ws.start()
    args = [{'channel': 'account', 'ccy': 'BTC'}]

    await ws.subscribe(args, callback=callbackFunc)
    await asyncio.sleep(10)

    await ws.unsubscribe(args, callback=callbackFunc)
    await asyncio.sleep(10)


asyncio.run(main())
