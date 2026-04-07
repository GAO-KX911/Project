from flask import Flask
from flask_sock import Sock

app = Flask(__name__)
sock = Sock(app)  # 绑定到同一个 Flask 实例

# HTTP 路由
@app.route('/')
def http_handler():
    return "This is HTTP"

# WebSocket 路由
@sock.route('/upload-pic')
def ws_handler(ws):
    ws.send("This is WebSocket")

app.run(port=8080)  # 同时监听 HTTP 和 WebSocket