import logging
import os
import argparse
import torch
import torch.optim as optim
from torch import nn
from torchvision import transforms, datasets, models
from tqdm import tqdm
from models.shuffle_psa import ShuffleNetV2_PSA
import torch.nn.functional as F
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, \
    classification_report
import matplotlib.pyplot as plt
import matplotlib


# 自定义ImageFolder类，返回图片路径
class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        # 获取图片和标签
        img, label = super().__getitem__(index)
        # 获取图片路径
        img_path = self.imgs[index][0]
        return img, label, img_path


def eval_model(net, validate_loader, device, val_num, class_names):
    # validate
    net.eval()
    val_acc = 0.0  # accumulate accurate number / epoch
    labels_value, predicted_value = [], []
    false_negative_images = []
    false_positive_images = []
    abnormal_class = 0
    with torch.no_grad():
        val_bar = tqdm(validate_loader)
        for i, (images, labels,img_paths) in enumerate(val_bar):
            images = images.to(device)
            labels_value.extend(labels.numpy())
            labels = labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            predicted_value.extend(predicted.cpu().numpy())
            val_acc += (predicted == labels).sum().item()

            # Identify false negatives and false positives
            for j in range(labels.size(0)):
                if predicted[j] != labels[j]:
                    if predicted[j] == abnormal_class and labels[j] != abnormal_class:
                        false_positive_images.append((img_paths[j], labels[j].item(), predicted[j].item()))
                    elif predicted[j] != abnormal_class and labels[j] == abnormal_class:
                        false_negative_images.append((img_paths[j], labels[j].item(), predicted[j].item()))

        val_accurate = val_acc / val_num

        precision = precision_score(labels_value, predicted_value, average='macro')
        recall = recall_score(labels_value, predicted_value, average='macro')
        f1 = f1_score(labels_value, predicted_value, average='macro')

        report = classification_report(labels_value, predicted_value, target_names=class_names, digits=5)

        return val_accurate, precision, recall, f1, report, false_negative_images,false_positive_images


def main(args):
    # device = args.device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    image_path = args.data_path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    # FLAME数据集
    # test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "Test"),
    #                                     transform=data_transform["val"])

    # load val dataset
    test_dataset = ImageFolderWithPaths(root=os.path.join(image_path, "Test"),
                                        transform=data_transform["val"])

    test_num = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1, shuffle=False,
                                              num_workers=4)

    # create model
    net = ShuffleNetV2_PSA(stages_repeats=[4, 8, 1], stages_out_channels=[24, 116, 232, 464, 128],
                           num_classes=args.num_classes)
    # load pretrain weights
    model_weight_path = args.weight_path
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    print("using {} weights.".format(model_weight_path))
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'), strict=False)
    net.to(device)

    test_acc, test_pre, test_rec, test_f1, test_report, false_negative_images, false_positive_images = eval_model(
        net, test_loader, device, test_num, args.class_names)

    print("test_acc: ", test_acc)
    print("test_pre: ", test_pre)
    print("test_rec: ", test_rec)
    print("test_f1: ", test_f1)
    # print("test_fpr: ", test_fpr)
    print("test_report: \n", test_report)

    # Output false negatives and false positives
    print("\nFalse Negative Images:")
    for img_path, true_label, pred_label in false_negative_images:
        print(f"Image: {img_path}, True Label: {args.class_names[true_label]}, Predicted Label: {args.class_names[pred_label]}")

    print("\nFalse Positive Images:")
    for img_path, true_label, pred_label in false_positive_images:
        print(f"Image: {img_path}, True Label: {args.class_names[true_label]}, Predicted Label: {args.class_names[pred_label]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_names', nargs='+', default=["abnormal", "neutral"])
    # 类别数量
    parser.add_argument('--num_classes', type=int, default=2)
    # 数据集的根目录
    parser.add_argument('--data_path', type=str, default='../DataSet_Njust_binary/')
    # 最佳权重的路径
    parser.add_argument('--weight_path', type=str,
                        default= './Our/Case1/Train_03/lr_0.006/best.pth',
                        help='initial weights path')


    opt = parser.parse_args()

    main(opt)  
