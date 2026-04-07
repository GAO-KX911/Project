import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pycocotools.mask as mask_util
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from ..utils import mask2ndarray

EPS = 1e-2


def color_val_matplotlib(color):
    """Convert various input in BGR order to normalized RGB matplotlib color
    tuples,

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[float]: A tuple of 3 normalized floats indicating RGB channels.
    """
    color = mmcv.color_val(color)
    color = [color / 255 for color in color[::-1]]
    return tuple(color)


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      segms=None,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      mask_color=None,
                      thickness=6,
                      font_size=26,
                      win_name='',
                      show=True,
                      wait_time=0,
                      out_file=None):
    assert bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes.shape[0] == labels.shape[0], \
        'bboxes.shape[0] and labels.shape[0] should have the same length.'
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    img = mmcv.imread(img).astype(np.uint8)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    mask_colors = []
    if labels.shape[0] > 0:
        if mask_color is None:
            # random color
            np.random.seed(42)
            mask_colors = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
        else:
            # specify  color
            mask_colors = [
                              np.array(mmcv.color_val(mask_color)[::-1], dtype=np.uint8)
                          ] * (
                                  max(labels) + 1)

    # bbox_color = color_val_matplotlib(bbox_color)
    # text_color = color_val_matplotlib(text_color)
    # 在这里修改颜色
    # bbox_color1 = color_val_matplotlib('green')
    bbox_color = bbox_color
    # bbox_color = [(0, 0, 255), (48, 124, 236)]
    class_names = ['flame', 'smoke']
    print('bbox_color:', bbox_color)
    print('thick,font:', thickness, font_size)
    bbox_color1 = color_val_matplotlib(bbox_color[0])
    bbox_color2 = color_val_matplotlib(bbox_color[1])
    # print(bbox_color1)
    # print(bbox_color2)
    # bbox_color2 = color_val_matplotlib('red')
    text_color1 = color_val_matplotlib(text_color[0])
    text_color2 = color_val_matplotlib(text_color[1])

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')


    # 修改
    polygonsA = []
    polygonsB = []
    color1 = []
    color2 = []

    # for i, (bbox, label) in enumerate(zip(bboxes, labels)):
    #     bbox_int = bbox.astype(np.int32)
    #     poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
    #             [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
    #     np_poly = np.array(poly).reshape((4, 2))
    #     polygons.append(Polygon(np_poly))
    #     color.append(bbox_color)
    #     label_text = class_names[
    #         label] if class_names is not None else f'class {label}'
    #     if len(bbox) > 4:
    #         label_text += f'|{bbox[-1]:.02f}'
    #     ax.text(
    #         bbox_int[0],
    #         bbox_int[1],
    #         f'{label_text}',
    #         bbox={
    #             'facecolor': 'black',
    #             'alpha': 0.8,
    #             'pad': 0.7,
    #             'edgecolor': 'none'
    #         },
    #         color=text_color,
    #         fontsize=font_size,
    #         verticalalignment='top',
    #         horizontalalignment='left')

    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        bbox_int = bbox.astype(np.int32)

        label_text = class_names[
            label] if class_names is not None else f'class {label}'

        if len(bbox) > 4:
            label_text += f':{bbox[-1]:.03f}'

        if 'flame' in label_text:
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            polygonsA.append(Polygon(np_poly))

            color1.append(bbox_color1)

            ax.text(
                bbox_int[0],
                # bbox_int[0]- 1.8*font_size,#exceed-our-1.5 #标注超过图片的右边界，向左移动
                bbox_int[1] - 1.4 * font_size,
                f'{label_text}',
                bbox={
                    'facecolor': bbox_color1,
                    'alpha': 0.8,
                    'pad': 0.7,
                    'edgecolor': 'none'
                },
                color='white',
                fontsize=font_size,
                verticalalignment='top',
                horizontalalignment='left')

        elif 'smoke' in label_text:
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            polygonsB.append(Polygon(np_poly))

            color2.append(bbox_color2)

            ax.text(
                bbox_int[0],
                # bbox_int[1] - 1.5 * font_size,
                # bbox_int[1] + 0.2 * font_size, #19442.jpg标注文字向下移动
                bbox_int[3] + 0.12 * font_size,# f_694.jpg,烟雾标签换到左下角
                f'{label_text}',
                bbox={
                    'facecolor': bbox_color2,
                    'alpha': 0.8,
                    'pad': 0.7,
                    'edgecolor': 'none'
                },
                color='white',
                fontsize=font_size,
                verticalalignment='top',
                horizontalalignment='left')

        if segms is not None:
            color_mask = mask_colors[labels[i]]
            mask = segms[i].astype(bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5

    plt.imshow(img)

    # p = PatchCollection(
    #     polygons, facecolor='none', edgecolors=color, linewidths=thickness)
    # ax.add_collection(p)
    pA = PatchCollection(
        polygonsA, facecolor='none', edgecolors=color1, linewidths=thickness)
    pB = PatchCollection(
        polygonsB, facecolor='none', edgecolors=color2, linewidths=thickness)
    ax.add_collection(pA)
    ax.add_collection(pB)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    plt.close()

    return img


# 修改标注文本向下移动或向左移动
"""
def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      segms=None,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      mask_color=None,
                      thickness=6,
                      font_size=26,
                      win_name='',
                      show=True,
                      wait_time=0,
                      out_file=None):

    assert bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes.shape[0] == labels.shape[0], \
        'bboxes.shape[0] and labels.shape[0] should have the same length.'
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    img = mmcv.imread(img).astype(np.uint8)

    # width = img.shape[1]  # 获取图像的宽度

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    mask_colors = []
    if labels.shape[0] > 0:
        if mask_color is None:
            # random color
            np.random.seed(42)
            mask_colors = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
        else:
            # specify  color
            mask_colors = [
                np.array(mmcv.color_val(mask_color)[::-1], dtype=np.uint8)
            ] * (
                max(labels) + 1)


    # 在这里修改颜色

    bbox_color = bbox_color

    class_names = ['fire', 'smoke']
    print('bbox_color:', bbox_color)
    print('thick,font:',thickness,font_size)
    bbox_color1 = color_val_matplotlib(bbox_color[0])
    bbox_color2 = color_val_matplotlib(bbox_color[1])

    text_color1 = color_val_matplotlib(text_color[0])
    text_color2 = color_val_matplotlib(text_color[1])

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    # 修改
    polygonsA = []
    polygonsB = []
    color1 = []
    color2 = []


    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        bbox_int = bbox.astype(np.int32)

        label_text = class_names[
            label] if class_names is not None else f'class {label}'

        if len(bbox) > 4:
            label_text += f':{bbox[-1]:.03f}'

        # # 使用Axes对象的transData属性来将数据坐标转换为显示坐标
        # trans = ax.transData
        #
        # # 使用text函数绘制文本，但不显示在图上
        # text_obj = ax.text(0, 0, f'{label_text}', bbox={
        #             'facecolor': bbox_color1,
        #             'alpha': 0.8,
        #             'pad': 0.7,
        #             'edgecolor': 'none'
        #         },
        #         color='white',
        #         fontsize=font_size,
        #         verticalalignment='top',
        #         horizontalalignment='left', visible=False)
        #
        # # 计算文本的宽度
        # text_bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
        # text_width = text_bbox.width # 1?
        # print("img_width:", width)
        # print("text_width:", text_width)

        if 'fire' in label_text:
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            polygonsA.append(Polygon(np_poly))

            color1.append(bbox_color1)

            print("before-bbox_int[0]", bbox_int[0])
            bbox_int[0] = min(bbox_int[0], width - 6.8*font_size)
            print("after-bbox_int[0]", bbox_int[0])

            ax.text(
                bbox_int[0],
                bbox_int[1] - 1.5*font_size,
                f'{label_text}',
                bbox={
                    'facecolor': bbox_color1,
                    'alpha': 0.8,
                    'pad': 0.7,
                    'edgecolor': 'none'
                },
                color='white',
                fontsize=font_size,
                verticalalignment='top',
                horizontalalignment='left')


        elif 'smoke' in label_text:
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            polygonsB.append(Polygon(np_poly))

            color2.append(bbox_color2)
            bbox_int[0] = min(bbox_int[0], width - 6.8*font_size)

            ax.text(
                bbox_int[0],
                # bbox_int[1] + 0.2 * font_size, #19442.jpg
                bbox_int[1] - 1.5*font_size,
                f'{label_text}',
                bbox={
                    'facecolor': bbox_color2,
                    'alpha': 0.8,
                    'pad': 0.7,
                    'edgecolor': 'none'
                },
                color='white',
                fontsize=font_size,
                verticalalignment='top',
                horizontalalignment='left')

        if segms is not None:
            color_mask = mask_colors[labels[i]]
            mask = segms[i].astype(bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5

    plt.imshow(img)

    # p = PatchCollection(
    #     polygons, facecolor='none', edgecolors=color, linewidths=thickness)
    # ax.add_collection(p)
    pA = PatchCollection(
        polygonsA, facecolor='none', edgecolors=color1, linewidths=thickness)
    pB = PatchCollection(
        polygonsB, facecolor='none', edgecolors=color2, linewidths=thickness)
    ax.add_collection(pA)
    ax.add_collection(pB)


    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    if show:
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    plt.close()

    return img
"""


def imshow_gt_det_bboxes(img,
                         annotation,
                         result,
                         class_names=None,
                         score_thr=0,
                         gt_bbox_color=(255, 102, 61),
                         gt_text_color=(255, 102, 61),
                         gt_mask_color=(255, 102, 61),
                         det_bbox_color=(72, 101, 241),
                         det_text_color=(72, 101, 241),
                         det_mask_color=(72, 101, 241),
                         thickness=2,
                         font_size=13,
                         win_name='',
                         show=True,
                         wait_time=0,
                         out_file=None):
    """General visualization GT and result function.

    Args:
      img (str or ndarray): The image to be displayed.)
      annotation (dict): Ground truth annotations where contain keys of
          'gt_bboxes' and 'gt_labels' or 'gt_masks'
      result (tuple[list] or list): The detection result, can be either
          (bbox, segm) or just bbox.
      class_names (list[str]): Names of each classes.
      score_thr (float): Minimum score of bboxes to be shown.  Default: 0
      gt_bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: (255, 102, 61)
      gt_text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: (255, 102, 61)
      gt_mask_color (str or tuple(int) or :obj:`Color`, optional):
           Color of masks. The tuple of color should be in BGR order.
           Default: (255, 102, 61)
      det_bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: (72, 101, 241)
      det_text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: (72, 101, 241)
      det_mask_color (str or tuple(int) or :obj:`Color`, optional):
           Color of masks. The tuple of color should be in BGR order.
           Default: (72, 101, 241)
      thickness (int): Thickness of lines. Default: 2
      font_size (int): Font size of texts. Default: 13
      win_name (str): The window name. Default: ''
      show (bool): Whether to show the image. Default: True
      wait_time (float): Value of waitKey param. Default: 0.
      out_file (str, optional): The filename to write the image.
         Default: None

    Returns:
        ndarray: The image with bboxes or masks drawn on it.
    """
    assert 'gt_bboxes' in annotation
    assert 'gt_labels' in annotation
    assert isinstance(
        result,
        (tuple, list)), f'Expected tuple or list, but get {type(result)}'

    gt_masks = annotation.get('gt_masks', None)
    if gt_masks is not None:
        gt_masks = mask2ndarray(gt_masks)

    img = mmcv.imread(img)

    img = imshow_det_bboxes(
        img,
        annotation['gt_bboxes'],
        annotation['gt_labels'],
        gt_masks,
        class_names=class_names,
        bbox_color=gt_bbox_color,
        text_color=gt_text_color,
        mask_color=gt_mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=False)

    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None

    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        segms = mask_util.decode(segms)
        segms = segms.transpose(2, 0, 1)

    img = imshow_det_bboxes(
        img,
        bboxes,
        labels,
        segms=segms,
        class_names=class_names,
        score_thr=score_thr,
        bbox_color=det_bbox_color,
        text_color=det_text_color,
        mask_color=det_mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=show,
        wait_time=wait_time,
        out_file=out_file)
    return img
