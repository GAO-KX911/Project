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


def eval_model(net, validate_loader, device, val_num, class_names):
    # validate
    net.eval()
    val_acc = 0.0  # accumulate accurate number / epoch
    labels_value, predicted_value = [], []
    with torch.no_grad():
        val_bar = tqdm(validate_loader)
        for i, (images, labels) in enumerate(val_bar):
            images = images.to(device)
            labels_value.extend(labels.numpy())
            labels = labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            predicted_value.extend(predicted.cpu().numpy())
            val_acc += (predicted == labels).sum().item()

        val_accurate = val_acc / val_num

        precision = precision_score(labels_value, predicted_value, average='macro')
        recall = recall_score(labels_value, predicted_value, average='macro')
        f1 = f1_score(labels_value, predicted_value, average='macro')

        report = classification_report(labels_value, predicted_value, target_names=class_names, digits=5)

        return val_accurate, precision, recall, f1, report


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

    # load val dataset
    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "Val"),
                                        transform=data_transform["val"])

    # FLAME数据集
    # test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "Test"),
    #                                     transform=data_transform["val"])

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

    test_acc, test_pre, test_rec, test_f1, test_report = eval_model(net, test_loader, device, test_num,
                                                                    args.class_names)

    print("test_acc: ", test_acc)
    print("test_pre: ", test_pre)
    print("test_rec: ", test_rec)
    print("test_f1: ", test_f1)
    # print("test_fpr: ", test_fpr)
    print("test_report: ", test_report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_names', default=["0", "1", "2"])  # default=["fire", "smoke", "neutral"]
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--data_path', type=str, default="../Datasets/Our")
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weight_path', type=str,
                        default='./Our/ShuffleNet/CELS/lr_0.008/best.pth',
                        help='initial weights path')

    # parser.add_argument('--class_names', default=["0", "1"])  # default=["fire", "smoke", "neutral"]
    # parser.add_argument('--num_classes', type=int, default=2)
    # parser.add_argument('--data_path', type=str, default="/home/mayi/wd/lmy/Classification/Datasets/FLAME")
    # # 预训练权重路径，如果不想载入就设置为空字符
    # parser.add_argument('--weight_path', type=str, default='./FLAME/ShuffleNet/CE/lr_0.05/best.pth',
    #                     help='initial weights path')

    # parser.add_argument('--class_names', default=["0", "1", "2"])  # default=["fire", "smoke", "neutral"]
    # parser.add_argument('--num_classes', type=int, default=3)
    # parser.add_argument('--data_path', type=str, default="/home/mayi/wd/lmy/Classification/Datasets/DeepQuestAI")
    # # 预训练权重路径，如果不想载入就设置为空字符
    # parser.add_argument('--weight_path', type=str, default='./DeepQuestAI/ShuffleNet/CE/lr_0.001_lrf_8e-2/best.pth',
    #                     help='initial weights path')

    opt = parser.parse_args()

    main(opt)
