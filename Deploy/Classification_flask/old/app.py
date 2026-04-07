import os
import io
import json
import time  # 新增时间模块
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from model import ShuffleNetV2_PSA

app = Flask(__name__)
CORS(app)  # 解决跨域问题

weights_path = "/home/mayi/wd/zzl/Classification/Shuffle_PSA/Our/ShuffleNet/CELS/lr_0.008/best.pth"
class_json_path = "./class_indices.json"
assert os.path.exists(weights_path), "weights path does not exist..."
assert os.path.exists(class_json_path), "class json path does not exist..."

# select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# create model
model = ShuffleNetV2_PSA(stages_repeats=[4, 8, 1], stages_out_channels=[24, 116, 232, 464, 128], num_classes=3).to(device)
# load model weights
model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
model.eval()

# load class info
json_file = open(class_json_path, 'rb')
class_indict = json.load(json_file)

my_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        raise ValueError("input file does not RGB image...")
    return my_transforms(image).unsqueeze(0).to(device)


def get_prediction(image_bytes):
    try:
        
        
        tensor = transform_image(image_bytes=image_bytes)
        
        start_time = time.time()  # 记录开始时间
        outputs = model(tensor)
        # 计算推理时间
        elapsed_time = round((time.time() - start_time) * 1000, 3)  # 转换为毫秒保留3位小数
        
        outputs = torch.softmax(outputs, dim=1)
        prediction = outputs.squeeze(0).cpu().numpy()
        
        
        
        template = "class:{:<15} probability:{:.3f}"
        index_pre = [(class_indict[str(index)], float(p)) for index, p in enumerate(prediction)]
        index_pre.sort(key=lambda x: x[1], reverse=True)
        text = [template.format(k, v) for k, v in index_pre]
        return_info = {"result": text, "inference_time": elapsed_time}  # 添加时间字段
        
    except Exception as e:
        return_info = {"result": [str(e)], "inference_time": 0}
    return return_info


@app.route("/predict", methods=["POST"])
@torch.no_grad()
def predict():
    image = request.files["file"]
    img_bytes = image.read()
    info = get_prediction(image_bytes=img_bytes)
    return jsonify(info)


@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("fire.html")


if __name__ == '__main__':
    app.run(host="10.88.72.248", port=5000)