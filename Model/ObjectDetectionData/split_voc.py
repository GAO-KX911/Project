import os
import random

# 设置划分比例
trainval_percent = 0.9       # train + val 占全部数据的 90%
train_percent = 8/9          # 在 train + val 中，train 占 8/9

# 数据集根目录，可修改
base_path = '/home/mayi/wd/wxz/ObjectDetectionData/fire_smoke_voc'
# VOC 格式数据集中存放标注文件（.xml）的目录
xmlfilepath = os.path.join(base_path, 'Annotations')

# 用于保存划分结果的 txt 文件目录
txtsavepath = os.path.join(base_path, 'ImageSets', 'Main')
# 如果目录不存在就创建
if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)


# 获取该目录下所有 xml 文件的文件名列表
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)         # 总的 xml 文件数（即图像数量）
list = range(num)            # 编号列表，例如 range(0, num)

# 计算划分数据的数量
tv = int(num * trainval_percent)    # train + val 的数量
tr = int(tv * train_percent)        # train 的数量

# 从所有数据中随机采样 train+val 的索引
trainval = random.sample(list, tv)
# 从 train+val 中随机采样 train 的索引
train = random.sample(trainval, tr)

# 保存划分结果的 4 个 txt 文件
ftrainval = open(os.path.join(txtsavepath, 'trainval.txt'), 'w')
ftest = open(os.path.join(txtsavepath, 'test.txt'), 'w')
ftrain = open(os.path.join(txtsavepath, 'train.txt'), 'w')
fval = open(os.path.join(txtsavepath, 'val.txt'), 'w')

# 遍历所有样本，根据索引判断属于哪个子集，并写入对应文件
for i in list:
    name = total_xml[i][:-4] + '\n'  # 去掉 .xml 后缀，只保留文件名
    if i in trainval:
        ftrainval.write(name)        # 写入 trainval.txt
        if i in train:
            ftrain.write(name)       # 同时属于 train
        else:
            fval.write(name)         # 否则属于 val
    else:
        ftest.write(name)            # 不在 trainval 中的，属于 test

# 关闭文件
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
