"""
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

# 优化后的预处理流程
def fast_preprocess(cv_image):
    # 使用OpenCV进行快速预处理
    # Resize并CenterCrop等效操作（直接缩放到模型输入尺寸）
    h, w = cv_image.shape[:2]
    scale = 256 / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    cv_image = cv2.resize(cv_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Center crop
    y = (new_h - 224) // 2
    x = (new_w - 224) // 2                     
    cv_image = cv_image[y:y+224, x:x+224]
    
    # 转换为Tensor并标准化
    tensor = transforms.functional.to_tensor(cv_image)
    tensor = transforms.functional.normalize(
        tensor, 
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    return tensor.unsqueeze(0).to(device)

@torch.no_grad()
def predict(image):
    try:
        # 推理计时
        start_inference = time.time()
        
        # 直接使用Gradio传入的numpy数组（RGB格式）
        tensor = fast_preprocess(image)
        
        # 模型推理
        outputs = model(tensor)
        total_time = round((time.time() - start_inference) * 1000, 3)
        
        # 后处理
        outputs = torch.softmax(outputs, dim=1)
        prediction = outputs.squeeze(0).cpu().numpy()
        
        # 结果格式化
        results = [
            f"class:{class_indict[str(idx)]:<15} probability:{prob:.3f}"
            for idx, prob in sorted(enumerate(prediction), 
                                   key=lambda x: x[1], 
                                   reverse=True)
        ]
        
        return {
            "result": results,
            "inference_time_ms": total_time
        }
        
    except Exception as e:
        return {
            "result": [f"Error: {str(e)}"],
            "inference_time_ms": 0
        }

# 创建Gradio界面
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=gr.JSON(label="Prediction Results"),
    title="火灾分类系统",
    allow_flagging="never"
)

# 启动应用
if __name__ == "__main__":
    interface.launch(server_name="10.88.72.248", server_port=5000)
"""

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
def predict(image):
    try:
        start_time = time.time()
        tensor = fast_preprocess(image)
        outputs = model(tensor)
        inference_time = round((time.time() - start_time) * 1000, 3)
        
        probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
        sorted_indices = np.argsort(-probs)
        
        # 构建分类结果表格
        result_table = [[class_indict[str(idx)], f"{prob:.3%}"] 
                        for idx, prob in zip(sorted_indices, probs[sorted_indices])]
        
        return inference_time, result_table
        
    except Exception as e:
        return 0, [["错误", str(e)]]

# 自定义CSS美化界面
css = """
.gradio-container {
    background: #f5f7fb;
}
.result-box {
    border: 2px solid #4a90e2;
    border-radius: 10px;
    padding: 20px;
    margin: 15px 0;
    background: white;
}
.result-title {
    text-align: center;
    color: #4a90e2;
    font-weight: bold;
    margin-bottom: 10px;
}
.title-container {
    text-align: center;
    margin: 20px 0;
}

"""

# 构建界面布局
with gr.Blocks(css=css) as demo:
    
    with gr.Column(elem_classes="title-container"):
        gr.Markdown("# 火灾检测分类系统")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="numpy", label="输入图像")
            submit_btn = gr.Button("开始检测", variant="primary")
        
        with gr.Column():
            with gr.Column(elem_classes="result-box"):
                gr.Markdown("### 推理时间", elem_classes="result-title")
                time_output = gr.Number(label="毫秒", precision=3)
            
            with gr.Column(elem_classes="result-box"):
                gr.Markdown("### 分类结果", elem_classes="result-title")
                result_table = gr.DataFrame(
                    headers=["类别", "概率"],
                    datatype=["str", "str"],
                    interactive=False,
                    elem_id="result-table"
                )
    
    submit_btn.click(
        fn=predict,
        inputs=img_input,
        outputs=[time_output, result_table]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="10.88.72.248",
        server_port=5000
    )