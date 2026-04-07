import os
import json
import time
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import gradio as gr
from model import ShuffleNetV2_PSA

# 配置路径
weights_path = "/home/mayi/wd/zzl/Classification/Shuffle_PSA/Our/ShuffleNet/CELS/lr_0.008/best.pth"
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

def fast_preprocess(cv_image):
    cv_image = cv2.resize(cv_image, (224, 224), interpolation=cv2.INTER_LINEAR)
    tensor = transforms.functional.to_tensor(cv_image)
    tensor = transforms.functional.normalize(
        tensor, 
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    return tensor.unsqueeze(0).to(device)

@torch.no_grad()
def predict(original_image, crop_data):
    try:
        # 解析裁剪坐标
        x1, y1, x2, y2 = map(int, json.loads(crop_data))
        # 执行裁剪
        cropped_img = original_image[y1:y2, x1:x2]
        if cropped_img.size == 0:
            raise ValueError("Invalid crop area")
        
        # 预处理和推理
        start_time = time.time()
        tensor = fast_preprocess(cropped_img)
        outputs = model(tensor)
        inference_time = round((time.time() - start_time) * 1000, 3)
        
        # 处理结果
        probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
        sorted_indices = np.argsort(-probs)
        
        return inference_time, [[class_indict[str(idx)], f"{prob:.3%}"] for idx, prob in zip(sorted_indices, probs[sorted_indices])]
        
    except Exception as e:
        return 0, [["错误", str(e)]]

# 自定义CSS和JS
custom_css = """
#crop-container { position: relative; border: 2px dashed #4a90e2; }
#original-image { max-width: 100%; }
#selection-box { 
    position: absolute; 
    border: 2px solid #ff0000;
    background: rgba(255,0,0,0.1);
    pointer-events: none;
}
"""

custom_js = """
function setupCrop() {
    let isDragging = false;
    let startX, startY;
    const container = document.getElementById('crop-container');
    const img = document.getElementById('original-image');
    const box = document.getElementById('selection-box');
    
    container.onmousedown = (e) => {
        isDragging = true;
        const rect = img.getBoundingClientRect();
        startX = e.clientX - rect.left;
        startY = e.clientY - rect.top;
        box.style.display = 'block';
    }
    
    container.onmousemove = (e) => {
        if (!isDragging) return;
        const rect = img.getBoundingClientRect();
        const currentX = e.clientX - rect.left;
        const currentY = e.clientY - rect.top;
        
        box.style.left = Math.min(startX, currentX) + 'px';
        box.style.top = Math.min(startY, currentY) + 'px';
        box.style.width = Math.abs(currentX - startX) + 'px';
        box.style.height = Math.abs(currentY - startY) + 'px';
    }
    
    container.onmouseup = () => {
        isDragging = false;
        const rect = img.getBoundingClientRect();
        const finalX = parseInt(box.style.left) + parseInt(box.style.width);
        const finalY = parseInt(box.style.top) + parseInt(box.style.height);
        
        const cropData = JSON.stringify([
            parseInt(box.style.left),
            parseInt(box.style.top),
            finalX,
            finalY
        ]);
        
        document.getElementById('crop-data').value = cropData;
    }
}

document.addEventListener('DOMContentLoaded', setupCrop);
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 火灾检测分类系统（手动裁剪版）")
    
    with gr.Row():
        with gr.Column():
            # 原图显示区域
            with gr.Column(elem_id="crop-container"):
                original_image = gr.Image(elem_id="original-image", type="numpy", label="原图")
                gr.HTML("<div id='selection-box'></div>")
            
            # 隐藏的裁剪数据存储
            crop_data = gr.Textbox(visible=False, elem_id="crop-data")
            
            # 操作按钮
            gr.Markdown("操作步骤：1.上传图片 2.按住鼠标拖拽选区 3.点击检测")
            with gr.Row():
                upload_btn = gr.UploadButton("上传图片", file_types=["image"])
                submit_btn = gr.Button("开始检测", variant="primary")
        
        with gr.Column():
            # 结果显示
            gr.Markdown("### 推理时间")
            time_output = gr.Number(label="毫秒", precision=3)
            gr.Markdown("### 分类结果")
            result_table = gr.DataFrame(headers=["类别", "概率"])
    
    # 事件处理
    upload_btn.upload(
        fn=lambda file: cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)[..., ::-1],
        inputs=[upload_btn],
        outputs=[original_image]
    )
    
    submit_btn.click(
        fn=predict,
        inputs=[original_image, crop_data],
        outputs=[time_output, result_table]
    )
    
    # 注入自定义JS
    demo.head = f"<script>{custom_js}</script>"

if __name__ == "__main__":
    demo.launch(
        server_name="10.88.72.248",
        server_port=5000
    )