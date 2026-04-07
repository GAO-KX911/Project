import logging
import os
import argparse
import torch
from torch import nn
from tqdm import tqdm
import time
from torchvision import transforms, datasets
from thop import profile
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from models.shuffle_psa import ShuffleNetV2_PSA


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("using {} device.".format(device))

    # create model
    net = ShuffleNetV2_PSA(stages_repeats=[4, 8, 1], stages_out_channels=[24, 116, 232, 464, 128],
                           num_classes=args.num_classes)
    # load pretrain weights
    model_weight_path = args.weight_path
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    print("using {} weights.".format(model_weight_path))
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'), strict=False)
    net.to(device)

    # 计算模型参数量
    input = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(net, inputs=(input,))

    print("FLOPs=", str(flops / 1e6) + '{}'.format("M"))
    print("Params=", str(params / 1e6) + '{}'.format("M"))

    # fps
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
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "Val"),
                                            transform=data_transform["val"])

    # FLAME
    # validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "Test"),
    #                                         transform=data_transform["val"])

    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=1, shuffle=True,
                                                  num_workers=4)

    print("using {} images for validation.".format(val_num))

    # 选择200张图像进行推理
    num_images = 1000
    # print("using {} images for FPS testing.".format(num_images))
    total_time = 0

    with torch.no_grad():
        for i, (input_data, target) in enumerate(validate_loader):
            if i >= num_images:
                break

            input_data = input_data.to(device)

            start_time = time.time()
            output = net(input_data)
            end_time = time.time()

            inference_time = end_time - start_time
            total_time += inference_time

    # 计算平均FPS
    average_fps = num_images / total_time

    # print(f"总推理时间: {total_time} 秒")
    print(f"平均FPS: {average_fps}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 数据集所在根目录
    # parser.add_argument('--class_names', default=["0", "1", "2"])  # default=["fire", "smoke", "neutral"]
    # parser.add_argument('--num_classes', type=int, default=3)
    # parser.add_argument('--data_path', type=str, default="/home/mayi/wd/lmy/Classification/Datasets/Our")
    # # 预训练权重路径，如果不想载入就设置为空字符
    # parser.add_argument('--weight_path', type=str,
    #                     default='./Our/ShuffleNet/CELS/lr_0.008/best.pth',
    #                     help='initial weights path')

    # parser.add_argument('--class_names', default=["0", "1"])  # default=["fire", "smoke", "neutral"]
    # parser.add_argument('--num_classes', type=int, default=2)
    # parser.add_argument('--data_path', type=str, default="/home/mayi/wd/lmy/Classification/Datasets/FLAME")
    # # 预训练权重路径，如果不想载入就设置为空字符
    # parser.add_argument('--weight_path', type=str,
    #                     default='./FLAME/ShuffleNet/CE/lr_0.01/best.pth',
    #                     help='initial weights path')

    parser.add_argument('--class_names', default=["0", "1", "2"])  # default=["fire", "smoke", "neutral"]
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--data_path', type=str, default="/home/mayi/wd/wxz/Classification/Dataset/")
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weight_path', type=str, default='/home/mayi/wd/wxz/Classification/Shuffle_PSA/Our/ShuffleNet/lr_0.006/best.pth',
                        help='initial weights path')



    opt = parser.parse_args()

    main(opt)
