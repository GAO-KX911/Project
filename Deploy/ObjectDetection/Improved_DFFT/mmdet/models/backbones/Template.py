import imutils
import numpy as np
import cv2
import glob
import torch

# device =
device = torch.device("cuda:0")  # 或者根据你的GPU设备选择合适的设备
# 定义模板图像文件夹路径
templates_folder = '/home/mayi/data/lmy/ObjectDetection/Improved_DFFT/Template_img/flame_clip/'

# 定义尺度列表
# scales = [1.0, 1.5]  # 可根据需求自定义尺度
# scales = [1.0]

# 定义旋转角度列表
# angles = [-35, 0, 35]  # 可根据需求自定义角度
# angles = [0]




def template_construct(filename_list, th, tw, threshold=0.75):
    processed_images = []

    # 加载目标图像
    for filename in filename_list:
        img_rgb = cv2.imread(filename) #ndarray(1080, 1920, 3)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)#ndarray(1080, 1920)
        new_rgb = img_rgb.copy()

        # 获取模板图像文件路径列表
        template_files = glob.glob(templates_folder + '*.jpg')

        # 创建一个全零的遮罩图像，将矩形框内的像素值设为1
        mask = np.zeros_like(img_gray)

        # 遍历模板图像文件列表
        for template_file in template_files:
            # 加载模板图像
            template = cv2.imread(template_file, 0)

            h, w = template.shape[:2]

            # 如果模板尺寸大于目标图像尺寸，则缩小为当前尺寸的一半
            while h > img_gray.shape[0] or w > img_gray.shape[1]:
                template = cv2.resize(template, (w // 2, h // 2))
                h, w = template.shape[:2]

            # 进行模板匹配
            res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

            # 设置匹配阈值
            # threshold = 0.75

            # 获取匹配位置
            loc = np.where(res >= threshold)

            for pt in zip(*loc[::-1]):
                bottom_right = (pt[0] + w, pt[1] + h)
                mask[pt[1]:bottom_right[1], pt[0]:bottom_right[0]] = 1

                # 绘制矩形框，保持矩形框内像素值不变
                # cv2.rectangle(img_rgb, pt, bottom_right, (0, 0, 255), 2)

        # 将矩形框外的像素值设为黑色
        new_rgb[mask == 0] = [0, 0, 0]

        # cv.resize(img, (width, height))
        # Resize the processed image to a consistent size
        processed_image = cv2.resize(new_rgb, (tw, th))# ndarray(800, 1216, 3)

        # Append the processed image to the list
        processed_images.append(processed_image)

    # Convert the list of processed images to a single NumPy array
    numpy_array_temp = np.array(processed_images)

    # Convert the NumPy array to a PyTorch tensor
    tensor_temp = torch.tensor(numpy_array_temp).permute(0, 3, 1, 2).to(torch.float16)  # Shape: (len(filename_list), 3, height, width)

    # Shape: (len(filename_list), 3, height, width)

    return tensor_temp.to(device)


