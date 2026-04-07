from argparse import ArgumentParser
import os
from mmdet.apis import inference_detector, init_detector
import cv2
from tqdm import tqdm
from PIL import Image
import os
import random
from PIL import Image, ImageDraw
import mmcv
import argparse
import numpy as np



def main(args):
    # config文件
    config_file = args.config_file
    # 训练好的模型
    checkpoint_file = args.checkpoint_file

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # 图片路径
    img_dir = args.img_dir
    object_save_dir = args.object_save_dir
    bg_save_dir = args.bg_save_dir
    score_thr = args.score_thr


    if not os.path.exists(object_save_dir):
        os.mkdir(object_save_dir)
    if not os.path.exists(bg_save_dir):
        os.mkdir(bg_save_dir)

    count = 0
    img_names = os.listdir(img_dir)
    # 对图像文件进行随机排序
    random.shuffle(img_names)

    # 选择一半的图片作为背景掩码
    num_background_images = len(img_names) // 2
    background_images = img_names[:num_background_images]

    for img_name in tqdm(img_names):
        img = os.path.join(img_dir, img_name)
        count += 1
        print('model is processing the {}/{} images.'.format(count, len(img_names)))

        result = inference_detector(model, img)
        if isinstance(result, tuple):
            bbox_result, _ = result
        else:
            bbox_result = result
        bboxes = np.vstack(bbox_result)
        if score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]

        image = Image.open(img)
        if img_name in background_images:
            # 创建一个与图像大小相同的黑色掩码
            # image = Image.open(img)
            draw = ImageDraw.Draw(image)
            # 示例：假设目标物体的 bounding boxes 为 [(x1, y1, x2, y2)]
            for bbox in bboxes:
                draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], fill=0)
            mask_file = os.path.splitext(img_name)[0] + "_mask.jpg"
            image.save(os.path.join(bg_save_dir, mask_file))
        else:
            # 创建一个与图像大小相同的黑色掩码
            # image = mmcv.imread(img)
            mask = Image.new("RGB", (image.size[0], image.size[1]), (0, 0, 0))
            # draw = ImageDraw.Draw(mask)
            # 对于目标物体掩码图片，根据目标检测结果将目标区域置为黑色
            # 这里需要使用 mmdetection 进行目标检测，并获取 bounding boxes
            # 示例：假设目标物体的 bounding boxes 为 [(x1, y1, x2, y2)]
            # 针对每个 bounding box，在掩码图像上将区域置为与原图一致
            for bbox in bboxes:
                bbox = bbox.astype(np.int32)
                roi = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                mask.paste(roi, (bbox[0], bbox[1]))
                # mask.paste(image[bbox[1]:bbox[3], bbox[0]:bbox[2]], (bbox[0], bbox[1]))

            # 保存掩码图片
            mask_file = os.path.splitext(img_name)[0] + "_mask.jpg"
            mask.save(os.path.join(object_save_dir, mask_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 权重和训练记录所保存的目录
    parser.add_argument('--config_file', type=str, default='./our_coco_time/resume/dfft_time.py')
    parser.add_argument('--checkpoint_file', type=str, default='./our_coco_time/resume/resume/epoch_52.pth')  # 自己需要的分类个数+1，如2+1
    parser.add_argument('--img_dir', type=str, default='/home/mayi/wd/lmy/Classification/Datasets/Our/smoke')
    parser.add_argument('--object_save_dir', type=str, default='/home/mayi/wd/lmy/Classification/Datasets/Our/smoke_aug')
    parser.add_argument('--bg_save_dir', type=str, default='/home/mayi/wd/lmy/Classification/Datasets/Our/no_aug')
    parser.add_argument('--score_thr', type=float, default=0.5)
    args = parser.parse_args()
    main(args)
