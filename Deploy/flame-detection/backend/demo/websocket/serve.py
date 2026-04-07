import asyncio
import websockets

async def hello(websocket, path):
    await websocket.send("Hello World!")

async def main():
    async with websockets.serve(hello, "localhost", 8765):
        await asyncio.Future()  # 永久运行

asyncio.run(main())