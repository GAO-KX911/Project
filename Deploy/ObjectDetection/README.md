# 第四章-基于局部形态知识匹配的早期火灾目标检测研究-实验

## 依赖环境
详见requirements.txt

安装依赖环境 (目前已经装好依赖环境，直接激活使用conda activate dfft/dino),dino只对应DINO方法

## 文件夹介绍：
- /Dataset：放置数据集；其中coco表示私人数据集Our_flame_smoke，fire_smoke_coco为公开数据集。

- /multi-scale_imgs_input 为需要图例的多尺度火灾图像

- /case_test 为案例分析对应结果

- 其余每个文件夹对应一个方法，其中Improved_DFFT为本研究所提出的方法

- AdaMixer文件夹下包含Faster RCNN,YOLOF和AdaMixer方法

## 每个文件夹下包含训练、测试文件

