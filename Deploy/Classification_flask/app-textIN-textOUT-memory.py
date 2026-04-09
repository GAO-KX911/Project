import gevent
from gevent import monkey
monkey.patch_all()  # 必须在最开始

import os
import sys

# 允许导入与当前应用平级的 CameraFeed_flask 包
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import io
import json
import time
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from flask import Flask, jsonify, request, render_template, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from model import ShuffleNetV2_PSA
from PIL import Image
from werkzeug.utils import secure_filename
import requests
import base64
from video_recorder import get_video_recorder

CAMERA_SERVICE_URL = os.getenv("CAMERA_SERVICE_URL", "http://localhost:5002")
CAMERA_BASE64_ENDPOINT = f"{CAMERA_SERVICE_URL}/camera/base64"
CAMERA_FIRE_PULSE_ENDPOINT = f"{CAMERA_SERVICE_URL}/camera/fire_pulse"
CAMERA_START_ENDPOINT = f"{CAMERA_SERVICE_URL}/camera/start"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))



app = Flask(__name__)
CORS(app)

# 可选：是否在分类服务进程里自动占用相机
AUTO_INIT_CAMERA = os.getenv("AUTO_INIT_CAMERA", "true").lower() == "true"
TEST_FIRE_PULSE = os.getenv("TEST_FIRE_PULSE", "false").lower() == "true"

# 启动websocket
sio = SocketIO(app, cors_allowed_origins='*', async_mode='gevent')

def trigger_fire_pulse(repeat=1, interval=0.6):
    """调用相机服务触发 IO 脉冲"""
    try:
        resp = requests.post(
            CAMERA_FIRE_PULSE_ENDPOINT,
            json={"repeat": repeat, "interval": interval},
            timeout=5
        )
        data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        if resp.ok and data.get("success"):
            print("[IO] fire_pulse sent via camera service")
        else:
            msg = data.get("message") if isinstance(data, dict) else resp.text
            print(f"[IO] fire_pulse API failed: {msg}")
    except Exception as e:
        print(f"[IO] fire_pulse API error: {e}")

if AUTO_INIT_CAMERA:
    try:
        resp = requests.get(CAMERA_START_ENDPOINT, timeout=3)
        if resp.ok:
            print("[IO] camera start requested via API")
        else:
            print(f"[IO] camera start API failed: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"[IO] camera init failed (AUTO_INIT_CAMERA): {e}")


if TEST_FIRE_PULSE:
    trigger_fire_pulse()


# 配置路径
# weights_path = "/home/mayi/wd/Classification/Shuffle_PSA/Our/ShuffleNet/CELS/lr_0.008/best.pth"
weights_path = os.getenv("CLASSIFIER_WEIGHTS_PATH", os.path.join(CURRENT_DIR, "models", "cls_env_A02.pth"))
class_json_path = os.getenv("CLASSIFIER_CLASS_JSON_PATH", os.path.join(CURRENT_DIR, "class_indices.json"))
assert os.path.exists(weights_path), "weights path does not exist..."
assert os.path.exists(class_json_path), "class json path does not exist..."

# 设备选择
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 模型加载  
model = ShuffleNetV2_PSA(
    stages_repeats=[4, 8, 1],
    stages_out_channels=[24, 116, 232, 464, 128],
    num_classes=2
).to(device)
model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
model.eval()

# 类别标签
with open(class_json_path, 'r') as f:
    class_indict = json.load(f)
label_to_idx = {label: int(idx) for idx, label in class_indict.items()}
abnormal_idx = label_to_idx.get("异常", 0)

# 优化后的预处理流程
def fast_preprocess(cv_image):
    """使用OpenCV进行快速预处理"""
    h, w = cv_image.shape[:2]
    scale = 256 / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    cv_image = cv2.resize(cv_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    y = (new_h - 224) // 2
    x = (new_w - 224) // 2
    cv_image = cv_image[y:y+224, x:x+224]
    tensor = transforms.functional.to_tensor(cv_image)
    tensor = transforms.functional.normalize(
        tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    return tensor.unsqueeze(0).to(device)


@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("fire-textIN-textOUT.html")


@sio.on('upload_image')
def handle_upload_image(data):
    global classifyThreshold
    image_data, roi_rect, monitorType, classifyThreshold = data['image'], data['roiRect'], data['monitorType'], float(data['classifyThreshold'])
    
    # 获取 apiInvokeInterval 并计算 fps
    api_invoke_interval = float(data.get('apiInvokeInterval', '100'))
    fps = 1000.0 / api_invoke_interval if api_invoke_interval > 0 else 10.0
    # 获取或创建 video_recorder（使用正确的 fps）, 默认3s没有火焰停止录制
    video_recorder = get_video_recorder(output_dir='./recordings', fps=fps, max_no_flame_frames=3*fps)
    # 重新开始检测时，允许开始新录制
    video_recorder.enable_recording(request.sid)

    if monitorType == 'remote_camera':
        try:
            response = requests.get(CAMERA_BASE64_ENDPOINT, timeout=3)
            response.raise_for_status()
            image_data = response.json().get("image", "")
        except Exception as e:
            print(f"[camera] 获取远程图像失败: {e}")
            emit('predict_result', {'error': '无法获取远程相机图像'}, room=request.sid)
            return
    else:
        image_data = image_data.split(',')[1]


    # gevent或者thread都是异步的，等消息emit时，会丢失上下文，导致消息无法发送到正确的客户端，默认采用广播给所有用户，那就乱了
    # 所以需要补充sid和room机制来标识当前用户的会话    
    gevent.spawn(predict, image_data, 'original', request.sid, video_recorder, monitorType)
    # 当大于阈值时, 再去调用detect模型
    # gevent.spawn(detect, image_data, 'original', request.sid)

    if roi_rect:
        x, y, width, height = roi_rect['x'], roi_rect['y'], roi_rect['width'], roi_rect['height']
        
        # 将字节数据转换为PIL Image对象
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        roi_image = image.crop((x, y, x + width, y + height))
        
        # 将ROI图像转换回base64格式
        roi_buffer = io.BytesIO()
        roi_image.save(roi_buffer, format='JPEG')
        roi_base64 = base64.b64encode(roi_buffer.getvalue()).decode('utf-8')

        gevent.spawn(predict, roi_base64, 'roi', request.sid, video_recorder, monitorType)  # pyright: ignore[reportUndefinedVariable]
        # gevent.spawn(detect, roi_base64, 'roi', request.sid)


    # 立即返回，不阻塞
    emit('upload_image_result', {'message': 'Image received successfully'}, room=request.sid)


@sio.on('stop_detection')
def handle_stop_detection():
    """停止检测时保存当前录制的视频"""
    sid = request.sid
    print(f"停止检测，保存视频: {sid}")
    # 停止录制并保存视频（不删除会话，因为可能还会重新开始检测）
    video_recorder = get_video_recorder()
    gevent.spawn(video_recorder.stop_all_recordings, sid)


@sio.on('disconnect')
def handle_disconnect():
    """客户端断开连接时清理资源"""
    sid = request.sid
    print(f"客户端断开连接: {sid}")
    # 清理该会话的录制资源
    video_recorder = get_video_recorder()
    gevent.spawn(video_recorder.cleanup_session, sid)


@app.route("/api/videos", methods=["GET"], strict_slashes=False)
def get_videos():
    """获取录制的视频列表"""
    limit = request.args.get('limit', 100, type=int)
    video_recorder = get_video_recorder()
    videos = video_recorder.get_video_list(limit=limit)
    return jsonify({
        "success": True,
        "videos": videos,
        "count": len(videos)
    })


@app.route("/api/videos/<filename>", methods=["GET"])
def get_video(filename):
    """获取视频文件用于播放（支持 Range 请求）& 文件下载"""
    from flask import Response
    import mimetypes
    
    # 安全检查：防止路径遍历攻击
    # secure_filename 的作用： 删除特殊字符：../、./、/等 只保留安全的文件名; 示例：../../../etc/passwd→ etcpasswd
    filename = secure_filename(filename)
    video_recorder = get_video_recorder()
    video_path = os.path.join(video_recorder.output_dir, filename)
    
    if not os.path.exists(video_path):
        return jsonify({"error": "视频不存在"}), 404

    # 获取文件大小
    file_size = os.path.getsize(video_path)
    
    # 核心：支持 Range 请求（视频流式播放必需），不可能一次把完整视频返回给前端，太卡了
    range_header = request.headers.get('Range', None)
    
    if range_header:
        # 解析 Range 头
        byte_start = 0
        byte_end = file_size - 1
        
        match = range_header.replace('bytes=', '').split('-')
        if match[0]:
            byte_start = int(match[0])
        if match[1]:
            byte_end = int(match[1])
        
        length = byte_end - byte_start + 1
        
        # 读取文件片段
        with open(video_path, 'rb') as f:
            f.seek(byte_start)
            data = f.read(length)
        
        # 返回 206 Partial Content
        response = Response(
            data,
            206,  # Partial Content
            {
                'Content-Range': f'bytes {byte_start}-{byte_end}/{file_size}',
                'Accept-Ranges': 'bytes',
                'Content-Length': str(length),
            },
            direct_passthrough=True
        )
        return response
    else:
        # 普通请求，即文件下载，返回完整文件
        response = send_file(video_path)
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Content-Length'] = str(file_size)
        return response


@app.route("/api/videos/<filename>", methods=["DELETE"])
def delete_video(filename):
    """删除视频文件"""
    # 安全检查：防止路径遍历攻击
    filename = secure_filename(filename)
    video_recorder = get_video_recorder()
    video_path = os.path.join(video_recorder.output_dir, filename)
    
    if not os.path.exists(video_path):
        return jsonify({"error": "视频不存在"}), 404
    
    # 检查文件是否在允许的目录内, 二重校验，毕竟删除操作风险大一点
    if not os.path.abspath(video_path).startswith(os.path.abspath(video_recorder.output_dir)):
        return jsonify({"error": "非法访问"}), 403
    
    try:
        os.remove(video_path)
        return jsonify({"success": True, "message": "视频删除成功"})
    except Exception as e:
        return jsonify({"error": f"删除失败: {str(e)}"}), 500


@app.route("/api/videos/<filename>/rename", methods=["POST"])
def rename_video(filename):
    """重命名视频文件"""
    # 安全检查：防止路径遍历攻击
    filename = secure_filename(filename)
    video_recorder = get_video_recorder()
    video_path = os.path.join(video_recorder.output_dir, filename)
    
    if not os.path.exists(video_path):
        return jsonify({"error": "视频不存在"}), 404
    
    data = request.get_json()
    new_name = data.get('new_name', '').strip()
    
    if not new_name:
        return jsonify({"error": "新文件名不能为空"}), 400
    
    # 确保新文件名安全
    new_name = secure_filename(new_name)
    
    # 保持文件扩展名（支持 .avi 和 .mp4）
    original_ext = os.path.splitext(filename)[1]  # 获取原文件扩展名
    if not new_name.endswith(('.mp4', '.avi')):
        # 如果新文件名没有扩展名，使用原文件的扩展名
        new_name += original_ext
        
    new_path = os.path.join(video_recorder.output_dir, new_name)
    
    # 检查新文件名是否已存在
    if os.path.exists(new_path):
        return jsonify({"error": "文件名已存在"}), 400
    
    try:
        os.rename(video_path, new_path)
        return jsonify({"success": True, "message": "重命名成功", "new_filename": new_name})
    except Exception as e:
        return jsonify({"error": f"重命名失败: {str(e)}"}), 500

# ==================== AI 处理函数 ====================
def predict(pic_base64, prediction_type, sid, video_recorder, monitorType):
    """预测函数"""
    # print(f"start predict at {time.time()}, pic length: {len(pic_base64) if pic_base64 else 0}")
    
    if not pic_base64:
        return
    
    try:
        # 解码base64图像数据        
        image_bytes = base64.b64decode(pic_base64)
        image = Image.open(io.BytesIO(image_bytes))
        # 转换为OpenCV格式
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        tensor = fast_preprocess(cv_image)
        
        start_inference = time.time()
        outputs = model(tensor)
        inference_time = round((time.time() - start_inference) * 1000, 3)
        
        outputs = torch.softmax(outputs, dim=1)

        # prediction = outputs.squeeze(0).cpu().numpy()
        # 安装新的依赖，导致部署时报错：Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
        # 修改为下面这句了
        prediction = outputs.squeeze(0).cpu().detach().numpy()
        
        results = [
            f"分类结果：{class_indict[str(idx)]:<15}  概率：{prob:.3f}"
            for idx, prob in sorted(enumerate(prediction), key=lambda x: x[1], reverse=True)
        ]
        
        # 通过WebSocket发送结果
        sio.emit('predict_result', {
            'results': results,
            'inference_time': inference_time,
            'prediction_type': prediction_type
        }, room=sid)

        abnormal_probability = float(prediction[abnormal_idx])
        has_abnormal = abnormal_probability >= classifyThreshold

        if has_abnormal:
            trigger_fire_pulse()

        # 录制视频：检测到异常时开始/继续录制
        if prediction_type == 'original' and monitorType in  ('remote_camera', 'local_camera'):
            gevent.spawn(video_recorder.add_frame, sid, pic_base64, has_abnormal)
        
        if has_abnormal:
            gevent.spawn(detect, pic_base64, prediction_type, sid)
        else: 
            width, height = image.size
            sio.emit('detect_result', {
                "dimensions": {"height": height, "width": width },
                "time": "",
                "detections":[],
                "detection_type": prediction_type
            }, room=sid)

    except Exception as e:
        print(f"Predict error: {e}")
        sio.emit('predict_result', {
            'error': str(e), 
            "prediction_type": prediction_type
        }, room=sid)

def detect(image, detection_type, sid):
    """检测函数"""
    response = requests.post(
        'http://localhost:7866/detect-wb',
        json={
            'image': image,
            'detection_type': detection_type
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        # 添加检测类型字段
        result['detection_type'] = detection_type
        sio.emit('detect_result', result, room=sid)
    else:
        sio.emit('detect_result', {
            'error': f'API call failed with status {response.status_code}', 
            "detection_type": detection_type
        }, room=sid)

# ==================== 主程序入口 ====================
if __name__ == '__main__':
    sio.run(app, host='0.0.0.0', port=5001, debug=True)
