import sys
import os
import cv2
import numpy as np
import time
import base64
import threading
import json
from datetime import datetime
from flask import Flask, Response, jsonify, render_template_string, request, send_file
import io
import glob
import atexit, signal, sys

sys.path.append("/opt/MVS/Samples/64/Python")

from MvImport.MvCameraControl_class import *
from MvImport.MvErrorDefine_const import *
from MvImport.PixelType_header import *

# === 1. 把 MvImport 加进 sys.path（按你的实际路径修改） ===
# 路径是：/opt/MVS/Samples/64/Python/MvImport
mvimport_path = "/opt/MVS/Samples/64/Python/MvImport"
if mvimport_path not in sys.path:
    sys.path.append(mvimport_path)

from MvCameraControl_class import *
from MvErrorDefine_const import *



# === Flask 应用初始化 ===
app = Flask(__name__)

# === 全局变量 ===
camera_controller = None
is_capturing = False
latest_frame = None
frame_lock = threading.Lock()
capture_count = 0
start_time = time.time()

# === 配置 ===
CONFIG = {
    'CAPTURE_DIR': 'captures',
    'MAX_CAPTURES': 1000,
    'JPEG_QUALITY': 80,
    'FRAME_RATE': 30,
    'AUTO_START': True
}

# === 海康相机控制类 ===
class HikCameraController:
    def __init__(self):
        self.cam = None
        self.is_connected = False
        self.buffer_size = 40 * 1024 * 1024
        self.p_buf = None
        self.frame_info = MV_FRAME_OUT_INFO_EX()
        self.device_info = None
        
    def initialize_camera(self):
        """初始化相机"""
        try:
            if self.cam:
                return True
                
            # 枚举设备
            print("开始枚举相机设备...")
            device_list = MV_CC_DEVICE_INFO_LIST()
            ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
            print(f"枚举结果: ret=0x{ret:x}, device_count={device_list.nDeviceNum}")
            
            
            if ret != MV_OK or device_list.nDeviceNum == 0:
                print("❌ 枚举失败或未找到设备")
                return False

            print(f"找到 {device_list.nDeviceNum} 台设备")
            
            # 选择第一台设备
            self.device_info = cast(device_list.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
            
            self.cam = MvCamera()
            ret = self.cam.MV_CC_CreateHandle(self.device_info)
            if ret != MV_OK:
                print(f"创建设备句柄失败: 0x{ret:x}")
                self.cam = None
                return False

            ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != MV_OK:
                print(f"打开设备失败: 0x{ret:x}")
                self.cam.MV_CC_DestroyHandle()
                self.cam = None
                return False

            # 设置相机参数
            self.cam.MV_CC_SetEnumValue("TriggerMode", 0)  # 连续采集
            self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)  # 手动曝光
            self.cam.MV_CC_SetFloatValue("ExposureTime", 10000.0)  # 曝光时间
            
            # 开始采集
            ret = self.cam.MV_CC_StartGrabbing()
            if ret != MV_OK:
                print(f"开始采集失败: 0x{ret:x}")
                self.cam.MV_CC_CloseDevice()
                self.cam.MV_CC_DestroyHandle()
                self.cam = None
                return False

            # 分配缓冲区
            self.p_buf = (c_ubyte * self.buffer_size)()
            self.is_connected = True
            print("相机初始化成功")
            return True
            
        except Exception as e:
            print(f"相机初始化异常: {e}")
            self.is_connected = False
            return False

    def get_frame(self):
        """获取一帧图像"""
        global latest_frame
        
        if not self.is_connected or not self.cam:
            return None

        try:
            st_frame_info = MV_FRAME_OUT_INFO_EX()
            ret = self.cam.MV_CC_GetOneFrameTimeout(self.p_buf, self.buffer_size, st_frame_info, 1000)
            
            if ret != MV_OK:
                return None

            # 转换图像数据
            img = self.frame_to_ndarray(self.p_buf, st_frame_info)
            if img is not None:
                with frame_lock:
                    latest_frame = img
            return img
            
        except Exception as e:
            print(f"获取帧失败: {e}")
            return None

    def frame_to_ndarray(self, p_data, frame_info):
        """将相机帧数据转成 numpy 数组"""
        width = frame_info.nWidth
        height = frame_info.nHeight
        pixel_type = frame_info.enPixelType
        frame_len = frame_info.nFrameLen

        buf = np.frombuffer(p_data, dtype=np.uint8, count=frame_len)

        if pixel_type == PixelType_Gvsp_Mono8:
            # 单通道转三通道用于显示
            img = buf.reshape((height, width))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img

        elif pixel_type in (PixelType_Gvsp_BayerRG8, PixelType_Gvsp_BayerGB8,
                          PixelType_Gvsp_BayerGR8, PixelType_Gvsp_BayerBG8):
            img_raw = buf.reshape((height, width))
            img_bgr = cv2.cvtColor(img_raw, cv2.COLOR_BAYER_RG2BGR)
            return img_bgr

        elif pixel_type == PixelType_Gvsp_BGR8_Packed:
            return buf.reshape((height, width, 3))

        else:
            print(f"不支持的像素格式: 0x{pixel_type:x}")
            return None

    def capture_image(self, save_path=None):
        """捕获并保存图像"""
        img = self.get_frame()
        if img is None:
            return None, "获取图像失败"

        if save_path:
            success = cv2.imwrite(save_path, img, [cv2.IMWRITE_JPEG_QUALITY, CONFIG['JPEG_QUALITY']])
            if success:
                return img, f"图像已保存: {save_path}"
            else:
                return img, "图像保存失败"
        else:
            return img, "图像捕获成功"

    def get_camera_info(self):
        """获取相机信息"""
        if not self.is_connected:
            return {"status": "disconnected"}
        
        try:
            # 检查 device_info 是否存在
            device_type = "Unknown"
            if hasattr(self, 'device_info') and self.device_info:
                device_type = "USB" if self.device_info.nTLayerType == MV_USB_DEVICE else "GigE"
            
            info = {
                "status": "connected",
                "device_type": device_type,
                "frame_count": capture_count,
                "uptime": time.time() - start_time
            }
            return info
        except Exception as e:
            print(f"获取相机信息错误: {e}")
            return {"status": "error", "error": str(e)}
    

    def set_exposure(self, exposure_time):
        """设置曝光时间"""
        if self.is_connected and self.cam:
            try:
                ret = self.cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_time))
                return ret == MV_OK
            except:
                return False
        return False

    def release(self):
        """释放相机资源"""
        global is_capturing
        try:
            if self.cam:
                self.cam.MV_CC_StopGrabbing()
                self.cam.MV_CC_CloseDevice()
                self.cam.MV_CC_DestroyHandle()
                self.cam = None
                print("相机资源已释放")   # 只有在实际释放时打印
            # 如果没有 self.cam 则不重复打印
            self.is_connected = False
            is_capturing = False
        except Exception as e:
            print(f"释放相机资源异常: {e}")

# === 初始化相机控制器 ===
camera_controller = HikCameraController()

# === 工具函数 ===
def cleanup_old_captures():
    """清理旧文件，保持最多 MAX_CAPTURES 个文件"""
    capture_dir = CONFIG['CAPTURE_DIR']
    if not os.path.exists(capture_dir):
        return
    
    files = glob.glob(os.path.join(capture_dir, "*.jpg"))
    files.sort(key=os.path.getmtime)
    
    if len(files) > CONFIG['MAX_CAPTURES']:
        for file in files[:-CONFIG['MAX_CAPTURES']]:
            try:
                os.remove(file)
                print(f"清理旧文件: {file}")
            except:
                pass

def get_system_stats():
    """获取系统统计信息"""
    global capture_count, start_time
    
    return {
        "uptime": time.time() - start_time,
        "capture_count": capture_count,
        "capture_dir_size": sum(os.path.getsize(os.path.join(CONFIG['CAPTURE_DIR'], f)) 
                               for f in os.listdir(CONFIG['CAPTURE_DIR']) 
                               if os.path.isfile(os.path.join(CONFIG['CAPTURE_DIR'], f))),
        "frame_rate": capture_count / (time.time() - start_time) if time.time() > start_time else 0
    }

# === Flask 路由 ===
@app.route('/')
def index():
    """主页"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>海康工业相机控制</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .card { background: #f5f5f5; padding: 20px; margin: 10px 0; border-radius: 5px; }
            .button { background: #007cba; color: white; padding: 10px 20px; text-decoration: none; border-radius: 3px; margin: 5px; }
            .button:hover { background: #005a87; }
            .status { padding: 10px; border-radius: 3px; margin: 5px 0; }
            .online { background: #d4edda; color: #155724; }
            .offline { background: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 海康工业相机控制界面</h1>
            
            <div class="card">
                <h3>📊 系统状态</h3>
                <div id="status">加载中...</div>
            </div>
            
            <div class="card">
                <h3>🎬 实时视频</h3>
                <a class="button" href="/camera/video_feed" target="_blank">📹 查看实时视频流</a>
                <a class="button" href="/camera/video_page" target="_blank">📺 视频播放页面</a>
            </div>
            
            <div class="card">
                <h3>🖼️ 图像捕获</h3>
                <a class="button" href="/camera/capture" target="_blank">📸 捕获单张图片</a>
                <a class="button" href="/camera/base64" target="_blank">🔗 获取Base64图像</a>
                <a class="button" href="/camera/capture_sequence?count=10">🎞️ 连续捕获10张</a>
            </div>
            
            <div class="card">
                <h3>⚙️ 相机控制</h3>
                <a class="button" href="/camera/start" target="_blank">🚀 启动相机</a>
                <a class="button" href="/camera/stop" target="_blank">🛑 停止相机</a>
                <a class="button" href="/camera/available" target="_blank">🔍 检查状态</a>
                <a class="button" href="/camera/settings" target="_blank">⚡ 相机设置</a>
            </div>
            
            <div class="card">
                <h3>📁 文件管理</h3>
                <a class="button" href="/camera/list_captures" target="_blank">📋 查看捕获列表</a>
                <a class="button" href="/camera/cleanup" target="_blank">🧹 清理旧文件</a>
            </div>
            
            <div class="card">
                <h3>📊 统计信息</h3>
                <a class="button" href="/camera/stats" target="_blank">📈 查看统计</a>
                <a class="button" href="/camera/logs" target="_blank">📝 查看日志</a>
            </div>
        </div>
        
        <script>
            // 实时更新状态
            function updateStatus() {
                fetch('/camera/status')
                    .then(response => response.json())
                    .then(data => {
                        const statusDiv = document.getElementById('status');
                        const statusClass = data.camera_connected ? 'status online' : 'status offline';
                        statusDiv.innerHTML = `
                            <div class="${statusClass}">
                                <strong>相机状态:</strong> ${data.camera_connected ? '✅ 已连接' : '❌ 未连接'} | 
                                <strong>运行时间:</strong> ${data.uptime ? Math.round(data.uptime) + '秒' : 'N/A'} | 
                                <strong>捕获数量:</strong> ${data.capture_count || 0}
                            </div>
                        `;
                    });
            }
            
            setInterval(updateStatus, 3000);
            updateStatus();
        </script>
    </body>
    </html>
    '''

def generate_frames():
    """生成视频流帧"""
    global is_capturing
    
    is_capturing = True
    frame_interval = 1.0 / CONFIG['FRAME_RATE']
    
    while is_capturing:
        try:
            start_time = time.time()
            
            img = camera_controller.get_frame()
            if img is not None:
                # 编码为 JPEG
                ret, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, CONFIG['JPEG_QUALITY']])
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            # 控制帧率
            elapsed = time.time() - start_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
                
        except Exception as e:
            print(f"生成帧异常: {e}")
            break

@app.route('/camera/video_feed')
def video_feed():
    """实时视频流"""
    if not camera_controller.is_connected:
        if not camera_controller.initialize_camera():
            return "相机不可用", 503
    
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera/video_page')
def video_page():
    """视频播放页面"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>实时视频流</title>
        <style>
            body { margin: 0; padding: 20px; background: #000; text-align: center; }
            img { max-width: 95%; max-height: 95vh; border: 2px solid #333; }
            .controls { margin: 10px; }
            .button { background: #007cba; color: white; padding: 10px 20px; text-decoration: none; margin: 5px; }
        </style>
    </head>
    <body>
        <div class="controls">
            <a class="button" href="/">返回主页</a>
            <a class="button" href="/camera/capture">捕获图片</a>
        </div>
        <img src="/camera/video_feed" alt="实时视频流">
    </body>
    </html>
    '''

@app.route('/camera/capture')
def capture():
    """捕获单张图片"""
    global capture_count
    
    if not camera_controller.is_connected:
        if not camera_controller.initialize_camera():
            return jsonify({"success": False, "message": "相机不可用"})
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"capture_{timestamp}.jpg"
    save_path = os.path.join(CONFIG['CAPTURE_DIR'], filename)
    
    # 创建保存目录
    os.makedirs(CONFIG['CAPTURE_DIR'], exist_ok=True)
    
    # 捕获图像
    img, message = camera_controller.capture_image(save_path)
    
    if img is not None:
        capture_count += 1
        cleanup_old_captures()  # 自动清理旧文件
        
        return jsonify({
            "success": True,
            "message": message,
            "filename": filename,
            "path": save_path,
            "timestamp": timestamp,
            "size": f"{img.shape[1]}x{img.shape[0]}",
            "capture_count": capture_count
        })
    else:
        return jsonify({
            "success": False,
            "message": message
        })

@app.route('/camera/capture_sequence')
def capture_sequence():
    """连续捕获多张图片"""
    count = request.args.get('count', 5, type=int)
    interval = request.args.get('interval', 1.0, type=float)
    
    results = []
    for i in range(count):
        result = capture()
        results.append(result.get_json())
        if i < count - 1:  # 不是最后一次等待
            time.sleep(interval)
    
    return jsonify({
        "success": True,
        "message": f"成功捕获 {count} 张图片",
        "results": results
    })

@app.route('/camera/base64')
def camera_base64():
    """获取Base64格式图像"""
    if not camera_controller.is_connected:
        if not camera_controller.initialize_camera():
            return jsonify({"success": False, "message": "相机不可用"})
    
    img = camera_controller.get_frame()
    if img is not None:
        # 编码为 JPEG 然后转为 base64
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, CONFIG['JPEG_QUALITY']])
        base64_data = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        return jsonify({"image": base64_data})
    else:
        return jsonify({
            "success": False,
            "message": "获取图像失败"
        })

@app.route('/camera/available')
def camera_available():
    """检测相机是否可用"""
    try:
        # 尝试枚举设备
        device_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
        
        device_count = device_list.nDeviceNum if ret == MV_OK else 0
        is_available = device_count > 0
        
        return jsonify({
            "available": is_available,
            "device_count": device_count,
            "camera_connected": camera_controller.is_connected,
            "message": f"找到 {device_count} 台设备" if is_available else "未找到相机设备",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "available": False,
            "error": str(e),
            "message": "检测相机状态时发生错误"
        })

@app.route('/camera/start')
def start_camera():
    """启动相机"""
    if camera_controller.initialize_camera():
        return jsonify({
            "success": True,
            "message": "相机启动成功",
            "camera_connected": True
        })
    else:
        return jsonify({
            "success": False,
            "message": "相机启动失败",
            "camera_connected": False
        })

@app.route('/camera/stop')
def stop_camera():
    """停止相机"""
    global is_capturing
    is_capturing = False
    camera_controller.release()
    
    return jsonify({
        "success": True,
        "message": "相机已停止",
        "camera_connected": False
    })

@app.route('/camera/status')
def camera_status():
    """获取相机状态"""
    try:
        stats = get_system_stats()
        camera_info = camera_controller.get_camera_info()
        
        return jsonify({
            "camera_connected": camera_controller.is_connected,
            "is_capturing": is_capturing,
            "camera_info": camera_info,
            "system_stats": stats,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "camera_connected": False,
            "is_capturing": False
        })

@app.route('/camera/settings')
def camera_settings():
    """相机设置页面"""
    return '''
    <h2>相机设置</h2>
    <form action="/camera/set_exposure" method="post">
        <label>曝光时间 (μs): </label>
        <input type="number" name="exposure" value="10000" min="100" max="100000">
        <input type="submit" value="设置">
    </form>
    <a href="/">返回主页</a>
    '''

@app.route('/camera/set_exposure', methods=['POST'])
def set_exposure():
    """设置曝光时间"""
    exposure_time = request.form.get('exposure', 10000, type=float)
    success = camera_controller.set_exposure(exposure_time)
    
    return jsonify({
        "success": success,
        "message": f"曝光时间设置为 {exposure_time}μs" if success else "设置失败",
        "exposure_time": exposure_time
    })

@app.route('/camera/list_captures')
def list_captures():
    """列出所有捕获的图片"""
    if not os.path.exists(CONFIG['CAPTURE_DIR']):
        return jsonify({"captures": [], "count": 0})
    
    files = glob.glob(os.path.join(CONFIG['CAPTURE_DIR'], "*.jpg"))
    files.sort(key=os.path.getmtime, reverse=True)
    
    captures = []
    for file in files[:50]:  # 只显示最近50个
        stat = os.stat(file)
        captures.append({
            "filename": os.path.basename(file),
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "url": f"/camera/download/{os.path.basename(file)}"
        })
    
    return jsonify({
        "captures": captures,
        "count": len(captures),
        "total_size": sum(os.path.getsize(f) for f in files)
    })

@app.route('/camera/download/<filename>')
def download_capture(filename):
    """下载捕获的图片"""
    filepath = os.path.join(CONFIG['CAPTURE_DIR'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    else:
        return "文件不存在", 404

@app.route('/camera/cleanup')
def cleanup_captures():
    """清理旧文件"""
    old_count = len(glob.glob(os.path.join(CONFIG['CAPTURE_DIR'], "*.jpg")))
    cleanup_old_captures()
    new_count = len(glob.glob(os.path.join(CONFIG['CAPTURE_DIR'], "*.jpg")))
    
    return jsonify({
        "success": True,
        "message": f"清理完成，删除了 {old_count - new_count} 个文件",
        "remaining_files": new_count
    })

@app.route('/camera/stats')
def camera_stats():
    """统计信息"""
    stats = get_system_stats()
    return jsonify(stats)

@app.route('/camera/logs')
def camera_logs():
    """查看日志（简化版）"""
    return jsonify({
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "capture_count": capture_count,
        "camera_connected": camera_controller.is_connected,
        "system_uptime": time.time() - start_time
    })

# === 启动和清理 ===
@app.before_first_request
def initialize():
    """在第一个请求前初始化"""
    print("Flask 应用初始化...")
    os.makedirs(CONFIG['CAPTURE_DIR'], exist_ok=True)
    
    if CONFIG['AUTO_START']:
        print("自动启动相机...")
        camera_controller.initialize_camera()

# @app.teardown_appcontext
# def close_camera(exception=None):
#     """应用关闭时清理资源"""
#     camera_controller.release()

def close_camera_on_exit():
    try:
        camera_controller.release()
    except Exception:
        pass

atexit.register(close_camera_on_exit)

def _handle_exit(signum, frame):
    close_camera_on_exit()
    sys.exit(0)

signal.signal(signal.SIGINT, _handle_exit)
signal.signal(signal.SIGTERM, _handle_exit)

# === 主程序 ===
if __name__ == '__main__':
    print("=== 海康工业相机 Flask 服务 ===")
    print("访问地址: http://127.0.0.1:5002")
    print("可用接口:")
    print("  GET /                         - 主页")
    print("  GET /camera/video_feed        - 实时视频流")
    print("  GET /camera/video_page        - 视频播放页面")
    print("  GET /camera/capture           - 捕获单张图片")
    print("  GET /camera/capture_sequence  - 连续捕获多张")
    print("  GET /camera/base64            - 获取Base64图像")
    print("  GET /camera/available         - 检查相机状态")
    print("  GET /camera/start             - 启动相机")
    print("  GET /camera/stop              - 停止相机")
    print("  GET /camera/status            - 获取相机状态")
    print("  GET /camera/settings          - 相机设置页面")
    print("  GET /camera/list_captures      - 查看捕获列表")
    print("  GET /camera/stats             - 统计信息")
    print("  GET /camera/logs              - 查看日志")
    
    app.run(host='0.0.0.0', port=5002, debug=True, threaded=True)