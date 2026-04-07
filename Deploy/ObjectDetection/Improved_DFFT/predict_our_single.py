from argparse import ArgumentParser
import os
from mmdet.apis import inference_detector, init_detector  # , show_result_pyplot
import cv2
from tqdm import tqdm
from PIL import Image


def show_result_pyplot(model, img, result, score_thr=0.5, fig_size=(15, 10)):
    """Visualize the detection results on the image.
    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module
    # img = model.show_result(img, result, score_thr=score_thr, show=False)
    img = model.show_result(
        img,
        result,
        score_thr=score_thr,
        show=False,
        # wait_time=wait_time,
        # win_name=title,
        # out_file=out_file,
        thickness=8,
        font_size=58,
        # font_size=32,
        bbox_color=[(0, 0, 255), (48, 124, 236)],
        text_color=[(0, 0, 255), (48, 124, 236)])

    return img



def main():
    # config文件
    config_file = './our_coco_time/resume/dfft_time.py'
    # 训练好的模型
    checkpoint_file = './our_coco_time/resume/resume/epoch_52.pth'  # epoch_52


    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # 图片路径
    img_dir = '../case_test/our'
    out_dir = './multi-scale_imgs_output/our/small_case'

    # img_dir = '../multi-scale_imgs_input_new/our'
    # out_dir = './multi-scale_imgs_output/our'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)


    # img_name = 'f_9793.jpg'
    img_name = 'f_2902.jpg'
    # f2902修改bbox_int[0] = min(bbox_int[0], width - 6.8*font_size)左移文本框
    img = os.path.join(img_dir, img_name)
    print(img)
    result = inference_detector(model, img)
    img_new = show_result_pyplot(model, img, result, score_thr=0.5)
    cv2.imwrite("{}/{}".format(out_dir, img_name), img_new)



if __name__ == '__main__':
    main()
