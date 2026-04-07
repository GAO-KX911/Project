import gevent
from gevent import monkey
monkey.patch_all()  # 必须在最开始

import os
import io
import json
import time
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from model import ShuffleNetV2_PSA
from PIL import Image
from werkzeug.utils import secure_filename
import requests
import base64

app = Flask(__name__)
CORS(app)

# 启动websocket
sio = SocketIO(app, cors_allowed_origins='*', async_mode='gevent')


# 配置路径
# weights_path = "/home/mayi/wd/Classification/Shuffle_PSA/Our/ShuffleNet/CELS/lr_0.008/best.pth"
weights_path = "/home/mayi/wd/zzl/Classification_flask/models/Classification_yayanhuo_0622.pth"
class_json_path = "./class_indices.json"
assert os.path.exists(weights_path), "weights path does not exist..."
assert os.path.exists(class_json_path), "class json path does not exist..."

# 设备选择
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 模型加载
model = ShuffleNetV2_PSA(
    stages_repeats=[4, 8, 1],
    stages_out_channels=[24, 116, 232, 464, 128],
    num_classes=3
).to(device)
model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
model.eval()

# 类别标签
with open(class_json_path, 'r') as f:
    class_indict = json.load(f)

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

    if (monitorType == 'remote_camera'):
        response = requests.get('http://localhost:5002/camera/base64')
        image_data = response.json()["image"]
    else:
        image_data = image_data.split(',')[1]

    # gevent或者thread都是异步的，等消息emit时，会丢失上下文，导致消息无法发送到正确的客户端，默认采用广播给所有用户，那就乱了
    # 所以需要补充sid和room机制来标识当前用户的会话
    gevent.spawn(predict, image_data, 'original', request.sid)
    # 当大于阈值时, 再去调用detect模型
    # gevent.spawn(detect, image_data, 'original', request.sid)

    if roi_rect:
        x, y, width, height = roi_rect['x'], roi_rect['y'], roi_rect['width'], roi_rect['height']
        
        # 将字节数据转换为PIL Image对象
        image_bytes = base64.b64decode(image_data)  
        image = Image.open(io.BytesIO(image_bytes))
        roi_image = image.crop((x, y, x + width, y + height))
        
        roi_buffer = io.BytesIO()
        roi_image.save(roi_buffer, format='JPEG')
        roi_base64 = base64.b64encode(roi_buffer.getvalue()).decode('utf-8')

        gevent.spawn(predict, roi_base64, 'roi', request.sid)
        # gevent.spawn(detect, roi_base64, 'roi', request.sid)


    emit('upload_image_result', {'message': 'Image received successfully'}, room=request.sid)

# prediction_type: 'original' | 'roi'
def predict(pic_base64, prediction_type, sid):
    """预测函数"""
    print(f"start predict at {time.time()}, pic length: {len(pic_base64) if pic_base64 else 0}")
    
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

        #  "0": "火焰", "1": "中性",、"2": "烟雾"
        if prediction[0] >= classifyThreshold or prediction[2] >= classifyThreshold:
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
        sio.emit('predict_result', {'error': str(e)}, room=sid)

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
        sio.emit('detect_result', {'error': f'API call failed with status {response.status_code}', "detection_type": detection_type}, room=sid)


if __name__ == '__main__':
    sio.run(app, host="0.0.0.0", port=5001)