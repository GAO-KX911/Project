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
    num_classes = 3
    class_name = ["flame", "neutral", "smoke"]


    # create model
    net = ShuffleNetV2_PSA(stages_repeats=[4, 8, 1], stages_out_channels=[24, 116, 232, 464, 128],
                           num_classes=num_classes)

    # load pretrain weights
    model_weight_path = './Our/ShffleNet/CELS/lr_0.008/best.pth'
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    print("using {} weights.".format(model_weight_path))
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'), strict=False)
    net.to(device)



    # predict_and_save_csv(net, device, data_dir, true_label, class_name, output_csv_path)

    data_dir = '../Datasets/Our/Train/no_smokefire'  # no_smokefire
    output_csv_path = './Predict/our_psa_train_pred.csv'
    true_label = 'neutral'
    predict_and_save_csv(net, device, data_dir, true_label, class_name, output_csv_path)
    print(f'Predictions saved to {output_csv_path}')
    #
    data_dir = '../Datasets/Our/Train/fire'  # no_smokefire
    output_csv_path = './Predict/our_psa_train_pred.csv'
    true_label = 'flame'
    predict_and_save_csv(net, device, data_dir, true_label, class_name, output_csv_path)
    print(f'Predictions saved to {output_csv_path}')

    data_dir = '../Datasets/Our/Train/smoke'  # no_smokefire
    output_csv_path = './Predict/our_psa_train_pred.csv'
    true_label = 'smoke'
    predict_and_save_csv(net, device, data_dir, true_label, class_name, output_csv_path)
    print(f'Predictions saved to {output_csv_path}')


    # val
    data_dir = '../Datasets/Our/Val/no_smokefire'  # no_smokefire
    output_csv_path = './Predict/our_psa_val_pred.csv'
    true_label = 'neutral'
    predict_and_save_csv(net, device, data_dir, true_label, class_name, output_csv_path)
    print(f'Predictions saved to {output_csv_path}')

    data_dir = '../Datasets/Our/Val/fire'  # no_smokefire
    output_csv_path = './Predict/our_psa_val_pred.csv'
    true_label = 'flame'
    predict_and_save_csv(net, device, data_dir, true_label, class_name, output_csv_path)
    print(f'Predictions saved to {output_csv_path}')

    data_dir = '../Datasets/Our/Val/smoke'  # no_smokefire
    output_csv_path = './Predict/our_psa_val_pred.csv'
    true_label = 'smoke'
    predict_and_save_csv(net, device, data_dir, true_label, class_name, output_csv_path)
    print(f'Predictions saved to {output_csv_path}')


# FLAME数据集
def main_flame():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 2
    class_name = ["flame", "neutral"]


    # create model
    net = ShuffleNetV2_PSA(stages_repeats=[4, 8, 1], stages_out_channels=[24, 116, 232, 464, 128],
                           num_classes=num_classes)

    # load pretrain weights
    model_weight_path = './Flame/ShuffleNet/CE/lr_0.05/best.pth'
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    print("using {} weights.".format(model_weight_path))
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'), strict=False)
    net.to(device)


    # predict_and_save_csv(net, device, data_dir, true_label, class_name, output_csv_path)

    data_dir = '../Datasets/FLAME/Train/No_Fire'  # no_smokefire
    output_csv_path = './Predict/flame_psa_train_pred.csv'
    true_label = 'neutral'
    predict_and_save_csv(net, device, data_dir, true_label, class_name, output_csv_path)
    print(f'Predictions saved to {output_csv_path}')

    data_dir = '../Datasets/FLAME/Train/Fire'  # no_smokefire
    output_csv_path = './Predict/flame_psa_train_pred.csv'
    true_label = 'flame'
    predict_and_save_csv(net, device, data_dir, true_label, class_name, output_csv_path)
    print(f'Predictions saved to {output_csv_path}')


    # val
    data_dir = '../Datasets/FLAME/Test/No_Fire'  # no_smokefire
    output_csv_path = './Predict/flame_psa_val_pred.csv'
    true_label = 'neutral'
    predict_and_save_csv(net, device, data_dir, true_label, class_name, output_csv_path)
    print(f'Predictions saved to {output_csv_path}')

    data_dir = '../Datasets/FLAME/Test/Fire'  # no_smokefire
    output_csv_path = './Predict/flame_psa_val_pred.csv'
    true_label = 'flame'
    predict_and_save_csv(net, device, data_dir, true_label, class_name, output_csv_path)
    print(f'Predictions saved to {output_csv_path}')

def main_deep():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 3
    class_name = ["flame", "neutral", "smoke"]


    # create model
    net = ShuffleNetV2_PSA(stages_repeats=[4, 8, 1], stages_out_channels=[24, 116, 232, 464, 128],
                           num_classes=num_classes)

    # load pretrain weights
    model_weight_path = './DeepQuestAI/ShffleNet/CE/lr_0.001_lrf_8e-2/best.pth'
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    print("using {} weights.".format(model_weight_path))
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'), strict=False)
    net.to(device)



    # predict_and_save_csv(net, device, data_dir, true_label, class_name, output_csv_path)

    data_dir = '../Datasets/DeepQuestAI/Train/Neutral'  # no_smokefire
    output_csv_path = './Predict/deep_psa_train_pred.csv'
    true_label = 'neutral'
    predict_and_save_csv(net, device, data_dir, true_label, class_name, output_csv_path)
    print(f'Predictions saved to {output_csv_path}')
    #
    data_dir = '../Datasets/DeepQuestAI/Train/Fire'  # no_smokefire
    output_csv_path = './Predict/deep_psa_train_pred.csv'
    true_label = 'flame'
    predict_and_save_csv(net, device, data_dir, true_label, class_name, output_csv_path)
    print(f'Predictions saved to {output_csv_path}')

    data_dir = '../Datasets/DeepQuestAI/Train/Smoke'  # no_smokefire
    output_csv_path = './Predict/deep_psa_train_pred.csv'
    true_label = 'smoke'
    predict_and_save_csv(net, device, data_dir, true_label, class_name, output_csv_path)
    print(f'Predictions saved to {output_csv_path}')


    # val
    data_dir = '../Datasets/DeepQuestAI/Val/Neutral'  # no_smokefire
    output_csv_path = './Predict/deep_psa_val_pred.csv'
    true_label = 'neutral'
    predict_and_save_csv(net, device, data_dir, true_label, class_name, output_csv_path)
    print(f'Predictions saved to {output_csv_path}')

    data_dir = '../Datasets/DeepQuestAI/Val/Fire'  # no_smokefire
    output_csv_path = './Predict/deep_psa_val_pred.csv'
    true_label = 'flame'
    predict_and_save_csv(net, device, data_dir, true_label, class_name, output_csv_path)
    print(f'Predictions saved to {output_csv_path}')

    data_dir = '../Datasets/DeepQuestAI/Val/Smoke'  # no_smokefire
    output_csv_path = './Predict/deep_psa_val_pred.csv'
    true_label = 'smoke'
    predict_and_save_csv(net, device, data_dir, true_label, class_name, output_csv_path)
    print(f'Predictions saved to {output_csv_path}')

if __name__ == '__main__':
    # main()
    # main_flame()
    main_deep()
