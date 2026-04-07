import asyncio
import websockets

async def hello():
    async with websockets.connect("ws://localhost:8765") as websocket:
        print(await websocket.recv())

asyncio.run(hello())