#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
后台 7x24 监控脚本：
- 直接通过 HikCameraController 采集帧
- 生产者/消费者模式解耦采集与推理
- 告警帧保存到磁盘，并写入 SQLite 数据库
"""
import base64
import json
import os
import queue
import sqlite3
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import requests
import torch

from model import ShuffleNetV2_PSA

sys_path_added = "/home/njust/Fire/Deploy/CameraFeed_flask"
import sys
if sys_path_added not in sys.path:
    sys.path.insert(0, sys_path_added)
from hk_camera import HikCameraController  # noqa: E402


# ================= 基础配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DB_PATH = "/home/njust/Fire/Deploy/monitor.db"
ALERT_IMAGE_DIR = os.path.join(PROJECT_DIR, "captures", "alerts")
CLASS_JSON = os.path.join(BASE_DIR, "class_indices.json")
WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "Classification_yayanhuo_0622.pth")

CAMERA_ID = "hk_camera_01"
CLASS_THRESHOLD = 0.6
QUEUE_MAX_SIZE = 200
LOG_INTERVAL = 500
DETECT_URL = None  # 若需要联动检测服务，填 "http://127.0.0.1:7866/detect-wb"
USE_VIDEO = True
VIDEO_PATH = "/home/njust/Fire/Model/视频/compress/output-h264.mp4"
LOOP_VIDEO = True  # 播完是否循环


# ================= 工具函数 =================
def ensure_env():
    os.makedirs(ALERT_IMAGE_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            captured_at TEXT NOT NULL,
            camera_id TEXT,
            class_label TEXT NOT NULL,
            probability REAL NOT NULL,
            image_path TEXT NOT NULL,
            extra TEXT
        );
        """
    )
    conn.commit()
    conn.close()


def load_class_map():
    with open(CLASS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    idx2label = {int(k): v for k, v in data.items()}
    label2idx = {v: k for k, v in idx2label.items()}
    return idx2label, label2idx


def build_preprocess(device, input_size=224, short_side=256):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def preprocess(frame_bgr):
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = float(short_side) / float(min(h, w))
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        y = (new_h - input_size) // 2
        x = (new_w - input_size) // 2
        img = img[y:y + input_size, x:x + input_size]
        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device, non_blocking=True)
        tensor = (tensor - mean) / std
        return tensor

    return preprocess


def load_model(device):
    model = ShuffleNetV2_PSA(
        stages_repeats=[4, 8, 1],
        stages_out_channels=[24, 116, 232, 464, 128],
        num_classes=3,
    ).to(device)
    state = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval().requires_grad_(False)
    return model


def save_alert_image(frame_bgr):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    date_dir = os.path.join(ALERT_IMAGE_DIR, ts[:8])
    os.makedirs(date_dir, exist_ok=True)
    path = os.path.join(date_dir, f"{ts}.jpg")
    cv2.imwrite(path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def insert_alert(conn, label, prob, image_path, extra=None):
    conn.execute(
        "INSERT INTO alerts (captured_at, camera_id, class_label, probability, image_path, extra) VALUES (?,?,?,?,?,?)",
        (
            datetime.utcnow().isoformat(),
            CAMERA_ID,
            label,
            float(prob),
            image_path,
            json.dumps(extra, ensure_ascii=False) if extra else None,
        ),
    )
    conn.commit()


# ================= 线程逻辑 =================
def capture_worker(camera, frame_queue, stop_event):
    while not stop_event.is_set():
        if USE_VIDEO:
            ret, frame = camera.read()
            if not ret:
                if LOOP_VIDEO:
                    camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                stop_event.set()
                break
        else:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.001)
                continue
        try:
            frame_queue.put(frame, timeout=0.1)
        except queue.Full:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                frame_queue.put(frame, timeout=0.1)
            except queue.Full:
                pass


def inference_worker(frame_queue, stop_event, device, model, preprocess, idx2label, label2idx):
    conn = sqlite3.connect(DB_PATH)
    fire_idx = label2idx.get("火焰")
    smoke_idx = label2idx.get("烟雾")
    count = 0
    start = time.perf_counter()

    while not stop_event.is_set() or not frame_queue.empty():
        try:
            frame = frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        tensor = preprocess(frame)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        count += 1
        fire_prob = outputs[fire_idx] if fire_idx is not None else 0.0
        smoke_prob = outputs[smoke_idx] if smoke_idx is not None else 0.0

        alert_label = None
        alert_prob = 0.0
        if fire_prob >= CLASS_THRESHOLD:
            alert_label = "火焰"
            alert_prob = fire_prob
        elif smoke_prob >= CLASS_THRESHOLD:
            alert_label = "烟雾"
            alert_prob = smoke_prob

        if alert_label:
            image_path = save_alert_image(frame)
            extra = {"inference_ms": round((t1 - t0) * 1000, 3)}
            if DETECT_URL:
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                payload = {
                    "image": base64.b64encode(buffer.tobytes()).decode("utf-8"),
                    "detection_type": "monitor",
                }
                try:
                    resp = requests.post(DETECT_URL, json=payload, timeout=5)
                    if resp.ok:
                        extra["detection"] = resp.json()
                except Exception as err:
                    extra["detection_error"] = str(err)
            insert_alert(conn, alert_label, alert_prob, image_path, extra)
            print(f"[ALERT] {alert_label} prob={alert_prob:.3f} -> {image_path}")

        if count % LOG_INTERVAL == 0:
            elapsed = time.perf_counter() - start
            fps = count / elapsed if elapsed > 0 else 0.0
            print(f"[Monitor] processed={count} | avg FPS={fps:.2f}")

        frame_queue.task_done()

    conn.close()


# ================= 主函数 =================
def main():
    ensure_env()
    idx2label, label2idx = load_class_map()

    if USE_VIDEO:
        camera = cv2.VideoCapture(VIDEO_PATH)
        if not camera.isOpened():
            print(f"❌ failed to open video: {VIDEO_PATH}")
            return
    else:
        camera = HikCameraController()
        if not camera.initialize_camera():
            print("❌ camera init failed")
            return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    model = load_model(device)
    preprocess = build_preprocess(device)

    frame_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
    stop_event = threading.Event()

    producer = threading.Thread(target=capture_worker, args=(camera, frame_queue, stop_event), daemon=True)
    consumer = threading.Thread(
        target=inference_worker,
        args=(frame_queue, stop_event, device, model, preprocess, idx2label, label2idx),
        daemon=True,
    )

    producer.start()
    consumer.start()
    print("✅ background monitor running (Ctrl+C to stop)")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹️ shutting down monitor ...")
    finally:
        stop_event.set()
        producer.join(timeout=3)
        consumer.join(timeout=3)
        if USE_VIDEO:
            camera.release()
        else:
            camera.release()
        print("✅ monitor stopped")


if __name__ == "__main__":
    main()
