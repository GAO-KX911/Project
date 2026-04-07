# -*- coding: utf-8 -*-
import gevent
from gevent import monkey
monkey.patch_all()  # 必须在最开始

from flask import Flask, request
from flask_socketio import SocketIO, emit
import base64
import io
from PIL import Image
import cv2
import numpy as np
import requests
import random

app = Flask(__name__)
# 启动websocket
sio = SocketIO(app, cors_allowed_origins='*', async_mode='gevent')
classifyThreshold = 0

# ==================== WebSocket 相关 ====================
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
        
        # 将ROI图像转换回base64格式
        roi_buffer = io.BytesIO()
        roi_image.save(roi_buffer, format='JPEG')
        roi_base64 = base64.b64encode(roi_buffer.getvalue()).decode('utf-8')

        gevent.spawn(predict, roi_base64, 'roi', request.sid)
        # gevent.spawn(detect, roi_base64, 'roi', request.sid)


    # 立即返回，不阻塞
    emit('upload_image_result', {'message': 'Image received successfully'}, room=request.sid)

# ==================== AI 处理函数 ====================
def predict(pic, prediction_type, sid):
    """预测函数"""
    if not pic:
        return
    
    try:
        # 解码base64图像数据        
        image_bytes = base64.b64decode(pic)
        image = Image.open(io.BytesIO(image_bytes))
        # 转换为OpenCV格式
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        fireProbability = random.random()
        smokeProbability = random.random()

        # 通过WebSocket发送结果
        sio.emit('predict_result',{
            "inference_time": random.random()*10,
            "results": [
                "分类结果：火焰               概率：" + str(fireProbability),
                "分类结果：烟雾               概率：" + str(smokeProbability),
                "分类结果：中性               概率：" + str(random.random())
            ],
            "prediction_type": prediction_type
        }, room=sid)

        if fireProbability >= classifyThreshold or smokeProbability >= classifyThreshold:
            gevent.spawn(detect, pic, prediction_type, sid)
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
    # 调用远程api  localhost:7866/detect-wb
    response = requests.post(
        'http://localhost:7866/detect-wb',
        json={
            'image': image,
            'type': detection_type
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