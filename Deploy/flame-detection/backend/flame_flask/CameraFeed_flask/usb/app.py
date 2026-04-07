from io import BytesIO
from flask import Flask, Response, jsonify, send_file
import cv2
import subprocess
import time
from threading import Timer
import threading
import base64


app = Flask(__name__)

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0
):  
    """
    ## 整个管道的数据流程：
    1. 摄像头捕获 → nvarguscamerasrc 从 CSI 摄像头获取原始数据
    2. 格式定义 → 指定为 NV12 格式的 1280x720 视频
    3. 硬件转换 → nvvidconv 进行 GPU 加速的格式转换和可选翻转
    4. 分辨率调整 → 转换为指定的显示分辨率和 BGRx 格式
    5. 格式标准化 → videoconvert 确保格式兼容性
    6. 最终输出 → 转换为 BGR 格式并通过 appsink 提供给应用程序
    这个管道专门为 NVIDIA Jetson 平台设计，充分利用了硬件加速能力，适合实时视频处理应用。
    """
    pipeline = (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
        f"format=NV12, framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={display_width}, height={display_height}, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! appsink"
    )
    print("[Pipeline] GStreamer:", pipeline)
    return pipeline

def detect_camera():
    """
    自动检测使用哪种摄像头（CSI or USB）
    """
    try:
        # 使用 v4l2-ctl 检查摄像头
        result = subprocess.run(["v4l2-ctl", "--list-devices"], stdout=subprocess.PIPE, text=True)
        output = result.stdout

        print("[Camera Detect] v4l2-ctl 输出:\n", output)

        if "imx219" in output or "tegra-capture-vi" in output:
            print("[Camera Detect] 识别为 Jetson MIPI CSI 摄像头（使用 GStreamer）")
            return "gstreamer"

        # 否则尝试 USB 摄像头
        print("[Camera Detect] 未检测到 MIPI 摄像头，尝试使用 USB 摄像头（/dev/video0）")
        return "usb"

    except Exception as e:
        print("[Camera Detect Error]", e)
        print("v4l-utils 未安装，将自动尝试访问usb摄像头")
        return "usb"  # 默认降级使用 USB 摄像头

# 初始化摄像头
def open_camera(cam_type):    
    if cam_type == "gstreamer":
        cam = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    else:
        cam = cv2.VideoCapture(0)
        # cv2默认是4:3的像素比例(640 * 480), 尝试设置分辨率为 16:9 (e.g., 1280x720, 640 * 360)
        # 注意：这取决于您的USB摄像头是否支持此分辨率
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    return cam

camera_type = detect_camera()
camera = open_camera(camera_type) # 耗时10s+

# 无访问，定时释放视像头, 否则其它进程无法访问摄像头
last_access_time = 0
auto_release_timer = None
AUTO_RELEASE_DELAY = 10  # 10秒后自动释放

def schedule_auto_release():
    """超过AUTO_RELEASE_DELAY不访问，自动释放摄像头，否则其它程序无法使用摄像头"""
    global auto_release_timer
    # 更新最后访问时间
    global last_access_time
    last_access_time = time.time()
    
    # 取消之前的定时器
    if auto_release_timer:
        auto_release_timer.cancel()
    
    # 设置新的定时器
    auto_release_timer = Timer(AUTO_RELEASE_DELAY, auto_release_camera)
    auto_release_timer.start()

def auto_release_camera():
    """自动释放摄像头"""
    global camera
    # 如果超过指定时间没有访问，则释放摄像头
    if time.time() - last_access_time >= AUTO_RELEASE_DELAY and camera is not None:
        camera.release()
        # 即便carema对象引用，后续调用carema.read()也连接不上了，不如释放节约内存空间
        camera = None
        print(f"[AUTO] 摄像头已自动释放（{AUTO_RELEASE_DELAY}秒无访问）")

camera_open_lock = threading.Lock()
def reopen_carema():
    """一段时间不访问，camera可能会被释放掉，所以需要再次打开"""
    global camera
    # 前端页面初始化时，video_feed和camera_available会同时执行这个耗时操作，所以必须得有线程锁控制
    with camera_open_lock:
        if camera is None: 
            camera = open_camera(camera_type)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success or frame is None:
            print("[ERROR] 摄像头帧读取失败")
            break
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        schedule_auto_release()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return '<h2>访问 <a href="/camera/video_feed">视频流</a>，或 <a href="/camera/capture">截图</a> </h2>'

@app.route('/camera/video_feed')
def video_feed():
    reopen_carema()
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera/capture')
def capture():
    reopen_carema()

    success, frame = camera.read()
    print("[CAPTURE] success:", success)
    if not success or frame is None:
        return jsonify({"error": "无法从摄像头获取图像"}), 500

    # 直接编码为 JPEG 格式，不保存到本地
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]  # 高质量 JPEG
    success, buffer = cv2.imencode('.jpg', frame, encode_param)
    
    # 转换为字节流并直接返回
    img_io = BytesIO(buffer.tobytes())
    img_io.seek(0)
    print("[CAPTURE] 图像已生成，直接返回")

    schedule_auto_release()
    return send_file(img_io, mimetype='image/jpeg', as_attachment=False)
    

@app.route('/camera/base64')
def camera_base64():
    reopen_carema()
    _, frame = camera.read()
    _, buffer = cv2.imencode('.jpg', frame)
    base64_data = base64.b64encode(buffer.tobytes()).decode('utf-8')
    return jsonify({"image": base64_data})

@app.route('/camera/available')
def camera_available():
    """检测服务器摄像头是否可用"""
    reopen_carema()
    if not camera.isOpened():
        return {
            "available": False,
            "message": "远程摄像头无法打开",
        }

    # 如果摄像头被其它进程占用，上面的camera.isOpened()会通过，但是当前程序无法访问摄像头 
    # 所以还需要尝试读取一帧来确认摄像头真正可用
    success, frame = camera.read()
    schedule_auto_release()
    if success and frame is not None:
        return {
            "available": True,
            "message": "摄像头可用",
        }
    else:
        return {
            "available": False,
            "message": "摄像头无法读取帧数据, 可能被其它进程占用。",
            "error": "Failed to read frame"
        }

@app.route('/camera/delay-release')
def camera_release():
    schedule_auto_release()
    return {
        "message": f"如果无新的访问，摄像头将会在{AUTO_RELEASE_DELAY}秒后释放",
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, threaded=True)
