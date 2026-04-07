from flask import Flask, Response, jsonify, send_file
import cv2
import os
import time

app = Flask(__name__)

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0
):
    pipeline = (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
        f"format=NV12, framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={display_width}, height={display_height}, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! appsink"
    )
    print("[Pipeline]", pipeline)
    return pipeline

# 打开摄像头
camera = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
if not camera.isOpened():
    print("[ERROR] 摄像头未打开")

def generate_frames():
    while True:
        success, frame = camera.read()
        print("[DEBUG] success:", success)
        if not success or frame is None:
            print("[ERROR] 摄像头帧读取失败")
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("[ERROR] 图像编码失败")
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return '<h2>访问 <a href="/video_feed">视频流</a>，或 <a href="/capture">截图</a></h2>'

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
    success, frame = camera.read()
    print("[CAPTURE] success:", success)
    if not success or frame is None:
        return jsonify({"error": "无法从摄像头获取图像"}), 500

    filename = f"capture_{int(time.time())}.jpg"
    filepath = os.path.join("/home/mayi/wd/client", filename)
    cv2.imwrite(filepath, frame)
    print(f"[CAPTURE] 图像已保存：{filepath}")
    return send_file(filepath, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
