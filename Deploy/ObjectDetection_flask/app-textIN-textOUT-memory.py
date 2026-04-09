from flask import Flask, request, jsonify, render_template
import cv2

import os
import os.path as osp
import sys
# 当前文件所在目录：OD_flask/
current_dir = osp.dirname(osp.abspath(__file__))
# 项目根目录 project_root/
project_root = osp.dirname(current_dir)
# 拼接 mmdet 路径：project_root/OD/New
mmdet_path = osp.join(project_root, 'ObjectDetection', 'Improved_DFFT')
# 加入到 sys.path
if mmdet_path not in sys.path:
    sys.path.insert(0, mmdet_path)

from mmdet.apis import init_detector, inference_detector
import time
import base64
from PIL import Image
from io import BytesIO
import numpy as np

app = Flask(__name__)

# 加载配置文件
config_file = os.getenv(
    "DETECTOR_CONFIG_PATH",
    osp.join(project_root, 'ObjectDetection', 'Improved_DFFT', 'our_coco_time', 'dfft_time.py')
)
checkpoint_file = os.getenv(
    "DETECTOR_WEIGHTS_PATH",
    osp.join(current_dir, 'models', 'obj_env_A.pth')
)
img_cache = '../cache/' 
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 类别映射字典
CLASS_MAP = {
    0: "fire",
    1: "smoke"
}

# 没成功, 直接写入base64图片报错 inference_detector(model, img_bgr)
@app.route('/detect-wb2', methods=['POST'])
def detect2():
    # 从 request 中获取 JSON 数据
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400
    base64_data = data['image']

    # 处理data URI格式 (data:image/jpeg;base64,xxxxx)
    if base64_data.startswith('data:image'):
        base64_data = base64_data.split(',')[1]
    
    # 解码base64
    image_bytes = base64.b64decode(base64_data)
    # 转换为PIL图像
    pil_image = Image.open(BytesIO(image_bytes))
    # 转换为numpy数组 (RGB格式)
    img_array = np.array(pil_image)
    
    # 如果是RGBA，转换为RGB
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    # 转换为BGR格式 (OpenCV格式)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    height, width = img_bgr.shape[:2]
    
    start_time = time.time()
    # 直接传入numpy数组, 不使用路径
    result = inference_detector(model, img_bgr)

    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000

    detections = []
    for class_id, class_results in enumerate(result):
        for res in class_results:
            if res[4] >= 0.23:  # confidence threshold
                x1, y1, x2, y2, conf = map(lambda x: round(x, 2), res.tolist())
                coordinates = [(x1, y1), (x2, y2)]
                detections.append({
                    "class": CLASS_MAP.get(class_id, "unknown"),
                    "coordinates": coordinates,
                    "confidence": conf
                })

    return {
        "dimensions": {"width": width, "height": height},
        "detections": detections,
        "time": f"{elapsed_time:.3f}"
    }


# 只是为了临时让程序跑通
@app.route('/detect-wb', methods=['POST'])
def detect():
    data = request.get_json()
    print('收到的数据:', data.keys())
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400
    base64_data, detection_type = data['image'], data['detection_type']
    image_path = detection_type + ".jpg"
    
    # 将base64数据保存为图片文件
    try:
        # 解码base64数据
        image_data = base64.b64decode(base64_data)
        
        # 保存为图片文件
        with open(image_path, 'wb') as f:
            f.write(image_data)
    except Exception as e:
        return jsonify({'error': f'Failed to save image: {str(e)}'}), 400

    # 从 request 中获取 JSON 数据
    if not os.path.exists(image_path):
        return {"error": f"文件不存在：{image_path}"}

    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"无法读取图片：{image_path}"}

    height, width = img.shape[:2]

    start_time = time.time()
    result = inference_detector(model, image_path)
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000

    detections = []
    for class_id, class_results in enumerate(result):
        for res in class_results:
            if res[4] >= 0.23:  # confidence threshold
                x1, y1, x2, y2, conf = map(lambda x: round(x, 2), res.tolist())
                coordinates = [(x1, y1), (x2, y2)]
                detections.append({
                    "class": CLASS_MAP.get(class_id, "unknown"),
                    "coordinates": coordinates,
                    "confidence": conf
                })

    return {
        "dimensions": {"width": width, "height": height},
        "detections": detections,
        "time": f"{elapsed_time:.3f}"
    }

@app.route('/')
def index():
    return render_template('index-textIN-textOUT.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7865)


