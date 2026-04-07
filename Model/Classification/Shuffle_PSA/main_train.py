import logging
import os
import argparse
import torch
import time
import torch.optim as optim
from torch import nn
from torchvision import transforms, datasets, models
from tqdm import tqdm
import torch.nn.functional as F
from datetime import datetime
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from torchvision import transforms, datasets
from models.shuffle_psa import ShuffleNetV2_PSA
import matplotlib.pyplot as plt
import matplotlib
from models.loss import CrossEntropyLabelSmooth
from thop import profile
import torch.optim.lr_scheduler as lr_scheduler
import math

matplotlib.use('agg')
plt_train_loss = []
plt_train_ac = []
plt_vaild_loss = []
plt_vaild_ac = []


def print_plot(train_plot, vaild_plot, train_text, vaild_text, ac, name, title):
    plt.cla()
    plt.title(title)
    plt.xlabel("epoch")
    x = [i for i in range(1, len(train_plot) + 1)]
    plt.plot(x, train_plot, label=train_text)
    plt.plot(x, vaild_plot, label=vaild_text)
    plt.legend()
    plt.savefig(name)


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_resume_checkpoint(checkpoint_path, net, optimizer, scheduler, epoch, best_acc,
                           best_epoch, best_report, history, args):
    checkpoint = {
        "epoch": epoch,
        "model_state": net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "best_report": best_report,
        "history": history,
        "args": vars(args),
    }
    torch.save(checkpoint, checkpoint_path)


def load_resume_checkpoint(checkpoint_path, net, optimizer, scheduler, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    required_keys = {"epoch", "model_state", "optimizer_state", "scheduler_state"}
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        raise ValueError(
            "resume checkpoint is incomplete, missing keys: {}".format(sorted(missing_keys))
        )

    net.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    move_optimizer_state_to_device(optimizer, device)
    return checkpoint


def train_model(net, train_loader, device, loss_function, train_num, optimizer):
    net.train()
    train_acc = 0.0
    train_loss = 0.0
    train_bar = tqdm(train_loader)
    labels_value = []
    predicted_value = []
    for step, (images, labels) in enumerate(train_bar):
        images = images.to(device)
        labels_value.extend(labels.numpy())
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        # predicted_value = predicted_value.append(predicted)
        predicted_value.extend(predicted.cpu().numpy())
        train_acc += (predicted == labels).sum().item()

    train_loss = train_loss / len(train_loader)
    train_accurate = train_acc / train_num

    return train_loss, train_accurate


def eval_model(net, validate_loader, device, loss_function, val_num, class_names):
    # validate
    net.eval()
    val_acc = 0.0  # accumulate accurate number / epoch
    val_loss = 0.0
    labels_value, predicted_value = [], []
    with torch.no_grad():
        val_bar = tqdm(validate_loader)
        for i, (images, labels) in enumerate(val_bar):
            images = images.to(device)
            # labels_value = labels_value.append(labels)
            labels_value.extend(labels.numpy())
            labels = labels.to(device)

            outputs = net(images)
            loss = loss_function(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            # predicted_value = predicted_value.append(predicted)
            predicted_value.extend(predicted.cpu().numpy())
            val_acc += (predicted == labels).sum().item()

        val_accurate = val_acc / val_num
        val_loss = val_loss / len(validate_loader)

        # labels_value, predicted_value = labels_value.cpu().numpy(), predicted_value.cpu().numpy()
        # 计算精确率、召回率和F1值
        precision = precision_score(labels_value, predicted_value, average='macro')
        recall = recall_score(labels_value, predicted_value, average='macro')
        f1 = f1_score(labels_value, predicted_value, average='macro')

        report = classification_report(labels_value, predicted_value, target_names=class_names, digits=5)

        return val_loss, val_accurate, precision, recall, f1, report


def main(args):
    # device = args.device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    nw = args.workers
    epochs = args.epochs
    save_path = args.save_path + '/lr_' + str(args.lr)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    latest_checkpoint_path = os.path.join(save_path, "latest_checkpoint.pth")
    best_checkpoint_path = os.path.join(save_path, "best_checkpoint.pth")
    metric_file_path = os.path.join(save_path, "train_val_metrics.txt")
    print("using {} device.".format(device))

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

    # load train dataset
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "Train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    # load val dataset
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "Val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # create model
    net = ShuffleNetV2_PSA(stages_repeats=[4, 8, 1], stages_out_channels=[24, 116, 232, 464, 128],
                           num_classes=args.num_classes)
    net.to(device)

    input = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(net, inputs=(input,))

    print("FLOPs=", str(flops / 1e6) + '{}'.format("M"))
    print("Params=", str(params / 1e6) + '{}'.format("M"))

    if args.loss_func == 'CrossEntropyLabelSmooth':
        # define loss function (criterion) and optimizer
        loss_function = CrossEntropyLabelSmooth(num_classes=args.num_classes, epsilon=0.1)
    else:
        loss_function = nn.CrossEntropyLoss()

    print("loss_function: ", loss_function)
    # optimizer = torch.optim.SGD(net.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    pg = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    start_epoch = 0
    best_acc = 0.0
    best_epoch = 0
    best_report = ""
    plt_train_loss = []
    plt_train_ac = []
    plt_vaild_loss = []
    plt_vaild_ac = []

    if args.resume:
        resume_path = args.resume
        assert os.path.exists(resume_path), "resume checkpoint {} does not exist.".format(resume_path)
        checkpoint = load_resume_checkpoint(resume_path, net, optimizer, scheduler, device)
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        best_epoch = checkpoint.get("best_epoch", 0)
        best_report = checkpoint.get("best_report", "")
        history = checkpoint.get("history", {})
        plt_train_loss = history.get("train_loss", [])
        plt_train_ac = history.get("train_acc", [])
        plt_vaild_loss = history.get("val_loss", [])
        plt_vaild_ac = history.get("val_acc", [])
        print("resuming training from epoch {} using checkpoint {}".format(start_epoch + 1, resume_path))
    else:
        model_weight_path = args.weights
        assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
        pre_weights = torch.load(model_weight_path, map_location='cpu')
        pre_dict = {k: v for k, v in pre_weights.items() if
                    k in net.state_dict() and net.state_dict()[k].numel() == v.numel()}
        missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)
        print("loaded warm start weights from {}".format(model_weight_path))

    file_mode = 'a' if start_epoch > 0 and os.path.exists(metric_file_path) else 'w'
    with open(metric_file_path, file_mode) as file:
        if file_mode == 'w':
            file.write(f"lr:{args.lr}\nTime\tEpoch\tTrain_Loss\tTrain_Acc\tVal_Loss\tVal_Acc\tVal_Pre\tVal_Rec\tVal_F1\n")

        for epoch in range(start_epoch, epochs):
            train_loss, train_acc = train_model(net, train_loader, device, loss_function, train_num, optimizer)
            scheduler.step()

            val_loss, val_acc, val_pre, val_rec, val_f1, val_report = eval_model(net, validate_loader, device,
                                                                                  loss_function, val_num,
                                                                                  args.class_names)
            print('train epoch[{}/{}]\tloss:{:.5f}\ttrain_acc: {:.5f}\tval_acc:{:.5f}'.format(epoch + 1, epochs,
                                                                                               train_loss, train_acc,
                                                                                               val_acc))

            if val_acc >= best_acc:
                best_acc = val_acc
                best_report = val_report
                best_epoch = epoch + 1
                torch.save(net.state_dict(), save_path + '/best.pth')

            time1 = "%s" % datetime.now()
            file.write(f"{time1}\t{epoch + 1}\t{train_loss:.5f}\t{train_acc:.5f}\t{val_loss:.5f}\t{val_acc:.5f}"
                       f"\t{val_pre:.5f}\t{val_rec:.5f}\t{val_f1:.5f}\n")
            file.flush()

            plt_train_ac.append(train_acc)
            plt_train_loss.append(train_loss)
            plt_vaild_ac.append(val_acc)
            plt_vaild_loss.append(val_loss)

            history = {
                "train_loss": plt_train_loss,
                "train_acc": plt_train_ac,
                "val_loss": plt_vaild_loss,
                "val_acc": plt_vaild_ac,
            }
            save_resume_checkpoint(
                latest_checkpoint_path,
                net,
                optimizer,
                scheduler,
                epoch,
                best_acc,
                best_epoch,
                best_report,
                history,
                args,
            )

            if val_acc >= best_acc:
                save_resume_checkpoint(
                    best_checkpoint_path,
                    net,
                    optimizer,
                    scheduler,
                    epoch,
                    best_acc,
                    best_epoch,
                    best_report,
                    history,
                    args,
                )

            print_plot(plt_train_loss, plt_vaild_loss, "train_loss", "vaild_loss", False, save_path + "/loss.png",
                       "train_loss and val_loss")
            print_plot(plt_train_ac, plt_vaild_ac, "train_acc", "vaild_acc", True, save_path + "/acc.png",
                       "train_acc and val_acc")

            if epoch == epochs - 1:
                torch.save(net.state_dict(), save_path + '/last_epoch.pth')

    print('Best epoch: {}'.format(best_epoch))
    print('Best val_acc: {:.5f}'.format(best_acc))
    print('Best report: {}'.format(best_report))
    with open(metric_file_path, 'a') as file:
        file.write(f"Best epoch: {best_epoch}\tBest val_acc: {best_acc:.5f}\nBest report: {best_report}\n")
    print('Finished Training!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_names', nargs='+', default=["abnormal", "neutral"])
    # 类别数量
    parser.add_argument('--num_classes', type=int, default=2)
    # 设置训练回合为4000（数据集训练4000次）
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--workers', type=int, default=16)


    parser.add_argument('--lr', type=float, default=0.006)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--loss_func', type=str, default="CrossEntropyLoss")

    # 数据集所在根目录
    parser.add_argument('--data_path', type=str, default="../DataSet/DataSet_Njust_binary_01")

    # 预训练权重路径
    parser.add_argument('--weights', type=str, default='./shufflenetv2_x1.pth',
                        help='initial weights path')
    parser.add_argument('--resume', type=str, default='',
                        help='path to a full training checkpoint for true resume')
    # parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    # 权重和训练记录所保存的目录，可以自定义路径
    parser.add_argument('--save_path', type=str, default="./Our/Binary/Train_01/")
    opt = parser.parse_args()
    main(opt)
