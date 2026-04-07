#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将 LabelMe 格式标注数据转换为 Pascal VOC 格式
"""

import argparse
import glob
import os
import os.path as osp
import sys
import imgviz
import labelme  # 用于读取 LabelMe 的 JSON 文件，需安装
# 命令行安装
# pip install labelme -i https://pypi.tuna.tsinghua.edu.cn/simple

# 生成 VOC XML 所需模块
try:
    import lxml.builder
    import lxml.etree
except ImportError:
    print("请先安装 lxml：pip install lxml")
    sys.exit(1)


def main():
    base_dir = "/home/njust/Fire/Data/ObjectionData"          #"D:/fire_smoke"   # 修改为当前数据所在的根目录
    input_dir = "/home/njust/Fire/Data/ObjectionData/json_file/Train"   #"D:/fire_smoke/json"  # 修改为 LabelMe 标注文件json所在的文件夹路径
    output_dir = "/home/njust/Fire/Data/ObjectionData/json_file/Voc/Train"  #"D:/fire_smoke_voc"   # 修改为 VOC 格式输出的文件夹路径

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="LabelMe 转 VOC 格式（无可视化）")
    parser.add_argument("--input_dir", type=str, default=input_dir, help="LabelMe 标注文件夹路径")
    parser.add_argument("--output_dir", type=str, default=output_dir, help="VOC 输出文件夹路径")
    parser.add_argument("--labels", type=str, default=os.path.join(base_dir, 'json_labels.txt'), help="包含类别的文本文件，每行一个标签")
    args = parser.parse_args()

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print("已创建输出目录:", args.output_dir)

        # 创建 VOC 所需目录结构
        os.makedirs(osp.join(args.output_dir, "JPEGImages"))
        os.makedirs(osp.join(args.output_dir, "Annotations"))


    # 读取标签文件
    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1，__ignore__ 为 -1，_background_ 为 0
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        elif class_id == 0:
            assert class_name == "_background_"
        class_names.append(class_name)
    class_names = tuple(class_names)

    # 遍历每个 LabelMe JSON 文件
    for filename in glob.glob(osp.join(args.input_dir, "*.json")):
        print("Generating dataset from:", filename)

        print(filename)


        try:
            label_file = labelme.LabelFile(filename=filename)
            

            base = osp.splitext(osp.basename(filename))[0]
            out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")
            out_xml_file = osp.join(args.output_dir, "Annotations", base + ".xml")

            # 解码图像数据并保存
            if label_file.imageData is None:
                print(f"{filename} 缺少 imageData，跳过")
                continue
            img = labelme.utils.img_data_to_arr(label_file.imageData)
            imgviz.io.imsave(out_img_file, img)

            # 构建 XML
            maker = lxml.builder.ElementMaker()
            xml = maker.annotation(
                maker.folder(),
                maker.filename(base + ".jpg"),
                maker.database(),
                maker.annotation(),
                maker.image(),
                maker.size(
                    maker.height(str(img.shape[0])),
                    maker.width(str(img.shape[1])),
                    maker.depth(str(img.shape[2])),
                ),
                maker.segmented(),
            )

            # 处理每个标注的 shape
            bboxes = []
            labels = []
            for shape in label_file.shapes:
                if shape["shape_type"] != "rectangle":
                    print(
                        "Skipping shape: label={label}, " "shape_type={shape_type}".format(
                            **shape
                        )
                    )
                    continue

                class_name = shape["label"]
                class_id = class_names.index(class_name)

                (xmin, ymin), (xmax, ymax) = shape["points"]
                # swap if min is larger than max.
                # LabelMe 中标注矩形时，坐标可能为浮点数，但目标检测的VOC要求整数像素坐标，因此需要做类型转换。
                xmin, xmax = sorted([int(xmin), int(xmax)])
                ymin, ymax = sorted([int(ymin), int(ymax)])

                bboxes.append([ymin, xmin, ymax, xmax])
                labels.append(class_id)

                xml.append(
                    maker.object(
                        maker.name(shape["label"]),
                        maker.pose(),
                        maker.truncated(),
                        maker.difficult(),
                        maker.bndbox(
                            maker.xmin(str(xmin)),
                            maker.ymin(str(ymin)),
                            maker.xmax(str(xmax)),
                            maker.ymax(str(ymax)),
                        ),
                    )
                )

            # 保存 XML 文件
            with open(out_xml_file, "wb") as f:
                f.write(lxml.etree.tostring(xml, pretty_print=True))
        except Exception as e:
            print(filename+"\t 异常")
            print(e)


if __name__ == "__main__":
    main()
