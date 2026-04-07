import sys
sys.path.insert(0, "/home/njust/Fire/Deploy/CameraFeed_flask")

import os
import time
import torch
import cv2
import numpy as np
from model import ShuffleNetV2_PSA
from hk_camera import HikCameraController  # 直接用相机控制类

DURATION = 3.0

def fast_preprocess(cv_image, device):
    h, w = cv_image.shape[:2]
    scale = 256 / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    cv_image = cv2.resize(cv_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    y = (new_h - 224) // 2
    x = (new_w - 224) // 2
    cv_image = cv_image[y:y+224, x:x+224]
    tensor = torch.tensor(cv_image.transpose(2,0,1)).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(device)

def load_model(device):
    model = ShuffleNetV2_PSA(
        stages_repeats=[4,8,1],
        stages_out_channels=[24,116,232,464,128],
        num_classes=2
    ).to(device)
    weights_path = os.getenv("CLASSIFIER_WEIGHTS_PATH", "./models/Classification_abnormal_neutral.pth")
    model.load_state_dict(torch.load(weights_path,
                                     map_location=device), strict=False)
    model.eval()
    return model

def main():
    cam = HikCameraController()
    if not cam.initialize_camera():
        print("camera init failed")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    print(f"直接相机链路测试 {DURATION}s ...")
    start = time.perf_counter()
    end_time = start + DURATION
    count = 0

    while time.perf_counter() < end_time:
        frame = cam.get_frame()
        print("raw frame shape:", frame.shape)

        if frame is None:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = fast_preprocess(rgb, device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        count += 1
        print(f"frame {count}: infer={(t1 - t0)*1000:.2f} ms")

    total = time.perf_counter() - start
    fps = count / total if total > 0 else 0.0
    print(f"直接链路: {count} frames / {total:.2f}s = {fps:.2f} FPS")

    cam.release()

if __name__ == "__main__":
    main()
