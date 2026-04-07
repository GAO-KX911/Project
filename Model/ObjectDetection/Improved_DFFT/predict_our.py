from argparse import ArgumentParser
import os
from mmdet.apis import inference_detector, init_detector  # , show_result_pyplot
import cv2
from tqdm import tqdm
from PIL import Image


'''推理图片
修改mmdetection类框架中的标注框颜色
predict_our.py的model.show_result(img, result, score_thr=score_thr, show=False)bbox_color

修改标注框的位置，以防显示不完全
mmdet.models.detectors.base.BaseDetector的show_result调用imshow_det_bboxes
→mmdet.core.visualization.image的def imshow_det_bboxes
python setup.py install'''
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
        thickness=2,
        # font_size=32,
        font_size=16, 
        bbox_color=[(0, 0, 255), (48, 124, 236)],
        text_color=[(0, 0, 255), (48, 124, 236)])

    return img

def predict_single(model, img_dir, img_name, out_dir):
    img = os.path.join(img_dir, img_name)
    print(img)
    result = inference_detector(model, img)
    img_new = show_result_pyplot(model, img, result, score_thr=0.23)
    cv2.imwrite("{}/{}".format(out_dir, img_name), img_new)


def predict_dir(model, img_dir, out_dir):
    img_names = os.listdir(img_dir)
    print(img_names)

    count = 0
    for img_name in tqdm(img_names):
        img = os.path.join(img_dir, img_name)
        print(img)
        count += 1
        print('model is processing the {}/{} images.'.format(count, len(img_names)))
        result = inference_detector(model, img)
        img_new = show_result_pyplot(model, img, result, score_thr=0.5)
        cv2.imwrite("{}/{}".format(out_dir, img_name), img_new)


def main():
    # 训练使用的配置文件路径
    config_file = './our_coco_time/dfft_time.py'
    # 训练好的模型（最佳权重）
    checkpoint_file = './our_coco_time/Objection_Train/latest.pth' 

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # 原始案例
    # img_dir = '../case_test_new/our'
    # out_dir = './multi-scale_imgs_output/our/small_case'

    # 新案例
    # img_dir = '../case_test_2/our'
    # out_dir = './multi-scale_imgs_output/our/small_case_2' # 'f_724.jpg','f_3106.jpg'

    # img_dir = '../multi-scale_imgs_input_new/our'
    # out_dir = './multi-scale_imgs_output/our'

    # 输入图片的文件夹
    img_dir = './multi-scale_imgs_input/our/Test_02'
    # 输出图片的文件夹
    out_dir = './multi-scale_imgs_output/our/Test_02'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # 需要预测的图片，放到输入文件夹中
    img_name = '4.png'
    #predict_single(model, img_dir, img_name, out_dir)

    predict_dir(model, img_dir, out_dir)



if __name__ == '__main__':
    main()
