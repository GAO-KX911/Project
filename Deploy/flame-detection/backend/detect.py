from flask import Flask, jsonify, request
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import random

app = Flask(__name__)


@app.route('/detect-wb', methods=['POST'])
def detect():
    # 从 request 中获取 JSON 数据
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400
    base64_data = data['image']
    
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
    print(height, width)
    

    return {
        "detections": [
          {
            "class": "fire",
            "confidence": round(random.random(), 2),
            "coordinates": [
              [
                random.uniform(100, 110),
                random.uniform(100, 110),
              ],
              [
                random.uniform(300, 310),
                random.uniform(350, 400),
              ]
            ]
          },
          {
            "class": "fire",
            "confidence": 0.95,
            "coordinates": [
              [
                random.uniform(0, 10),
                random.uniform(0, 10),
              ],
              [
                random.uniform(100, 110),
                random.uniform(350, 400),
              ]
            ]
          },
          {
            "class": "smoke",
            "confidence": 0.85,
            "coordinates": [
              [
                random.uniform(200, 210),
                random.uniform(100, 110),
              ],
              [
                random.uniform(400, 640),
                random.uniform(350, 400),
              ]
            ]
          }
        ],
        "dimensions": {"height": height,"width": width,},
        "time": "279.153"
      }

@app.route('/')
def index():
    return 'hello world'


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7866)



