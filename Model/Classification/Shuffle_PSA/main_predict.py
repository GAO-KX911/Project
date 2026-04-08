import os
import time

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from models.shuffle_psa import ShuffleNetV2_PSA


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PREDICT_BATCH_SIZE = 64
PREDICT_NUM_WORKERS = 8


def list_images(data_dir):
    return sorted(
        file_name
        for file_name in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, file_name))
        and os.path.splitext(file_name)[1].lower() in VALID_IMAGE_EXTS
    )


class TestImageDataset(Dataset):
    def __init__(self, test_root, class_names, transform):
        self.transform = transform
        self.samples = []
        for true_label in class_names:
            data_dir = os.path.join(test_root, true_label)
            if not os.path.isdir(data_dir):
                continue
            for img_name in list_images(data_dir):
                img_path = os.path.join(data_dir, img_name)
                self.samples.append((img_path, img_name, true_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, img_name, true_label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)
        return image_tensor, img_path, img_name, true_label


def predict_testset(model, device, test_root, class_names, batch_size=64, num_workers=8):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = TestImageDataset(test_root, class_names, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    predictions = []
    inference_times_ms = []
    print("found {} test images".format(len(dataset)))

    for image_tensor, img_paths, img_names, true_labels in tqdm(dataloader, desc="Predicting Test", unit="batch"):
        image_tensor = image_tensor.to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.inference_mode():
            output = model(image_tensor)
            predict = torch.softmax(output, dim=1)
            pred_indices = torch.argmax(predict, dim=1).cpu().tolist()
        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_inference_ms = (time.perf_counter() - start_time) * 1000
        per_image_ms = batch_inference_ms / len(pred_indices)

        for img_path, img_name, true_label, pred_idx in zip(img_paths, img_names, true_labels, pred_indices):
            pred_label = class_names[pred_idx]
            inference_times_ms.append(per_image_ms)
            predictions.append({
                "img_name": img_name,
                "img_path": img_path,
                "true_label": true_label,
                "predict_id": pred_idx,
                "predict_cls": pred_label,
                "is_false_positive": true_label == "neutral" and pred_label == "abnormal",
                "inference_ms": round(per_image_ms, 4),
            })

    predictions_df = pd.DataFrame(predictions)
    predictions_df = predictions_df[
        ["img_name", "img_path", "true_label", "predict_id", "predict_cls", "is_false_positive", "inference_ms"]
    ]
    average_inference_ms = round(sum(inference_times_ms) / len(inference_times_ms), 4) if inference_times_ms else 0.0
    false_positive_df = predictions_df[predictions_df["is_false_positive"]]
    test_image_count = len(predictions_df)
    neutral_count = len(predictions_df[predictions_df["true_label"] == "neutral"])
    false_positive_rate = round(len(false_positive_df) / test_image_count, 6) if test_image_count > 0 else 0.0

    summary_rows = [{
        "test_image_count": test_image_count,
        "neutral_image_count": neutral_count,
        "false_positive_count": len(false_positive_df),
        "false_positive_rate": false_positive_rate,
        "average_inference_ms": average_inference_ms,
    }]
    summary_df = pd.DataFrame(summary_rows)
    false_positive_df = false_positive_df[
        ["img_name", "img_path", "true_label", "predict_cls", "inference_ms"]
    ].reset_index(drop=True)
    return predictions_df, summary_df, false_positive_df


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 2
    class_names = ["abnormal", "neutral"]
    model_weight_root = "./Our/Binary/Train_01_resume94/"
    data_root = "../DataSet/DataSet_Njust_binary_01/"
    output_csv_path = "./Predict/pred_test01.csv"
    summary_csv_path = "./Predict/pred_test01_summary.csv"
    false_positive_csv_path = "./Predict/pred_test01_false_positive.csv"

    net = ShuffleNetV2_PSA(
        stages_repeats=[4, 8, 1],
        stages_out_channels=[24, 116, 232, 464, 128],
        num_classes=num_classes,
    )

    model_weight_path = model_weight_root + "lr_0.006/best.pth"
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    print("using {} weights.".format(model_weight_path))
    net.load_state_dict(torch.load(model_weight_path, map_location="cpu"), strict=False)
    net.to(device)

    test_root = os.path.join(data_root, "Test")
    assert os.path.isdir(test_root), "{} path does not exist.".format(test_root)

    predictions_df, summary_df, false_positive_df = predict_testset(
        net,
        device,
        test_root,
        class_names,
        batch_size=PREDICT_BATCH_SIZE,
        num_workers=PREDICT_NUM_WORKERS,
    )
    predictions_df.to_csv(output_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    false_positive_df.to_csv(false_positive_csv_path, index=False)

    print("test predictions saved to {}".format(output_csv_path))
    print("test summary saved to {}".format(summary_csv_path))
    print("test false positive details saved to {}".format(false_positive_csv_path))


if __name__ == "__main__":
    main()
