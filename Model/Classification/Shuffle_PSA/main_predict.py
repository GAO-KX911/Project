import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from models.shuffle_psa import ShuffleNetV2_PSA


def predict_and_save_csv(model, device, data_dir, true_label, class_name, output_csv):
    model.eval()

    transform = transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    predictions_list = []

    for img_name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_name)
        print(img_name)
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            # output = model(image_tensor)
            # _, preds = torch.max(output, 1)
            # print(output, output.shape, img_name, preds, preds.item(), class_name[preds.item()])

            output = torch.squeeze(model(image_tensor)).cpu()
            predict = torch.softmax(output, dim=0)
            preds = torch.argmax(predict).numpy()
            print(img_name, output, output.shape, preds, class_name[preds.item()])
            # print(output, output.shape, img_name, preds, preds.item(), class_name[preds.item()])
            # # print(class_name[preds.item()])
            # return

            # output = torch.squeeze(model(image_tensor)).cpu()
            # predict = torch.softmax(output, dim=0)
            # preds = torch.argmax(predict).numpy()

        predictions_list.append({'img_name': img_name, 'true_label': true_label, 'predict_id': preds.item(), 'predict_cls': class_name[preds.item()]})
        # predictions_list.append({'img_name': img_name, 'predict_id': preds.item(),
        #                          'predict_cls': class_name[preds.item()]})

    predictions_df = pd.DataFrame(predictions_list)
    if os.path.exists(output_csv):
        # Append to existing file
        predictions_df.to_csv(output_csv, mode='a', header=False, index=False)
    else:
        # Create a new file
        predictions_df.to_csv(output_csv, index=False)

    # predictions_df.to_csv(output_csv, index=False)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 类别数量及映射
    num_classes = 2
    class_name = ["abnormal", "neutral"]
    # 模型权重根路径
    model_weight_root = './Our/ShuffleNet_test03/'
    # 数据集根路径
    data_root = '../DataSet_Njust_binary/'
    # 预测结果存储路径
    output_csv_path = './Predict/pred_test03.csv'


    # create model
    net = ShuffleNetV2_PSA(stages_repeats=[4, 8, 1], stages_out_channels=[24, 116, 232, 464, 128],
                           num_classes=num_classes)

    # 训练权重路径
    model_weight_path = model_weight_root + 'lr_0.006/best.pth'
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    print("using {} weights.".format(model_weight_path))
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'), strict=False)
    net.to(device)



    # predict_and_save_csv(net, device, data_dir, true_label, class_name, output_csv_path)

    for split_name in ("Train", "Val", "Test"):
        for class_dir, true_label in (("abnormal", "abnormal"), ("neutral", "neutral")):
            data_dir = os.path.join(data_root, split_name, class_dir)
            if not os.path.isdir(data_dir):
                continue
            predict_and_save_csv(net, device, data_dir, true_label, class_name, output_csv_path)
            print(f'{split_name}/{class_dir} predictions saved to {output_csv_path}')


if __name__ == '__main__':
    main()

    
