# 本研究所提方法-LMKMFD


### 文件及文件夹介绍

- our_coco_time文件夹：  存放对于自建数据集Our_flame_smoke的训练好的最优模型权重和指标记录文件

- kaggle_temp文件夹：  存放对于kaggle数据集的训练好的最优模型权重和指标记录文件

- Template_img文件夹：our数据集上的局部形态知识模版

- Template_img_kaggle文件夹：kaggle数据集上的局部形态知识模版

- log_file文件夹： 存放训练时的日志文件

- mmdet文件夹： 该框架的核心文件，提供调用接口，及数据集的接口，方便自己训练时候修改使用。
注意：修改该文件夹中的文件后均需要运行python setup.py install重新载入模型
```
apis ：训练和测试相关依赖的函数，有随机种子生成、训练、单GPU测试、多GPU测试等
core：内核，包括锚框生成、边界框计算、结果评估、数据结构、掩码生成、可视化、钩子函数等等核心代码
datasets ：数据加载器的具体实现，对应 configs/datasets
models ：不同模型的具体实现，分为不同的主干、颈部、头部、损失函数等等，对应 configs/models
    - backbones主要用于特征提取，集成了大部分骨架网络。
    # 改进时修改了DFFTNet_time.py文件，对不同数据集训练时需要改变该文件中的模板路径。
    # 改完后运行python setup.py install
    - detectors常见目标检测器
    - necks 主要负责对 backbone 的特征进行高效融合和增强，能够对输入的单尺度或者多尺度特征进行融合、增强输出等，在models/necks文件下实现的neck：
    - dense_heads负责模型的输出，得到最终的结果，不同算法的head模块复杂程度不同，灵活度高，目标检测算法输出一般包括分类和框坐标回归两个分支。目标检测网络的两种主要类型是one_stage和two_stage,one-stage 需要 DenseHead， two-stage 需要RoIHead，分别在dense_heads和roi_heads中。
    - roi_heads负责模型的输出，得到最终的结果，two-stage 需要RoIHead，分别在dense_heads和roi_heads中。
    - loss一般是分类和回归loss，对head输出的预测值和bbox encoder得到的targets进行剃度下降迭代训练。
utils ：通用工具
__init__.py：判断配置的mmcv是否符合要求
version.py ：记录mmdetection 的版本
```

- configs文件夹：该文件夹是很重要也是很常用的一个文件夹，里面包括了使用该框架支持的所有算法。
```
该文件夹下存放所有的配置文件，可以自己写一个配置文件，也可以继承自_base_下的四种配置文件
_base_/datasets : 数据集加载，数据集格式获取数据。
_base_/models ：不同的目标检测模型
_base_/schedules ：训练计划
_base_/default_runtime.py ：运行时的配置，主要配置权重保存频率、日志频率、日志等级等信息
dfft/：dfft的配置。
```

- docker文件夹： 如果使用的是docker方式安装，才用得到。

- docs文件夹：该框架的相关介绍文档及语言支持。

- requirements文件夹：运行该项目所需要的环境及库文件支持。

- resources文件夹：该框架介绍的一些图片展示。

- tests文件夹：该框架的一些测试代码，包括对数据集、模型、运行时间、混淆矩阵的测试。

- tools文件夹：该框架的使用接口，包括模型训练代码和模型测试代码，以及作图日志分析工具等。
```
analysis_tools：分析日志和预测效果
dataset_converters：数据集转换
deployment：部署工具
misc：杂项：下载数据集、打印配置信息等工具
test.py：测试模型效果
train.py：根据配置文件进行训练
```
- multi-scale_imgs_output文件夹：在两个数据集上进行多尺度测试的推理结果。

- case_test_output_new文件夹：在两个数据集上进行小尺度案例分析测试的推理结果。

- predict_our.py: our数据集中火灾图片的推理文件

- predict_our_single.py: our数据集中批量火灾图片的推理文件

- predict_kaggle.py: kaggle数据集中火灾图片的推理文件

- predict_kaggle_single.py: kaggle数据集中批量火灾图片的推理文件

- image.py：提取JSON文件

- setup.py: 模型载入文件，修改了mmdet中的文件后，需要运行python setup.py install重新载入模型

- dfft_medium_pretrained.pth  预训练的权重文件

### 训练、测试和推理流程-参考https://blog.csdn.net/qq_35077107/article/details/124768392
目前参数均已修改好，如需直接训练测试推理，则只需要1,6,7,8步骤

1. 激活环境并载入模型
```
conda activate dfft
python setup.py install
```

2. 数据集准备：mmdet的数据集支持 coco格式和 voc 格式, 但 voc 格式官方只自带了少量网络模型文件, 所以推荐使用 coco 格式的数据集
```angular2html
数据集制作和转换参考：https://blog.csdn.net/Evan_qin_yi_quan/article/details/131450992
```
3. 修改mmdet/core/evalution/class_names.py和mmdet/datasets/coco.py中的标签为自建数据集的类别
```
# class_names.py 修改
def coco_classes():
    return [
        'fire', 'smoke'
    ]
# coco.py 修改
CLASSES = ('fire', 'smoke') # 即修改为自己的目标类别，注意与制作的coco类型数据集中的类别标签对应。

# 修改完 class_names.py 和 coco.py 之后一定要重新编译代码，否则验证输出仍然为原类别，且训练过程中指标异常, 在根目录下执行命令重新编译:
python setup.py install
```

4. 简单运行训练命令, 生成配置文件
```
python tools/train.py configs/dfft/dfft.py --work-dir work_dirs
# 其中 work_dirs 为你的工作目录,训练产生的日志,模型,网络结构文件会存放于此

如 python tools/train.py configs/dfft/dfft.py --work-dir our_coco_time
运行完命令后，会生成一个包含所有配置信息的配置文件在./our_pre文件夹下面
```

5. 修改工作目录(如./our_coco_time)下生成的模型配置文件 dfft_time.py 的的相关参数
```
(1) 修改num_classes变量
全局搜索num_classes，将其值改为所使用数据集的类别数（注意不包含背景）。把搜索到的num_classes全改掉。

(2) 修改数据加载部分的信息
搜索 data_root, 先修改数据文件的根节点目录,然后依次修改下面代码中的训练集, 验证集, 测试集的数据集路径位置, 举例训练集如下:

(3) 修改训练图片大小, 训练时的 batch_size, 学习率, epoch, work_dir等

- 图片输入大小修改主要修改 img_scale, 修改为你的图片实际输入大小;

- 对于 batch_size, 主要由 GPU 数量与 samples_per_gpu 参数决定
workers_per_gpu: 读取数据时每个gpu分配的线程数 。一般设置为 2即可
samples_per_gpu: 每个gpu读取的图像数量，该参数和训练时的gpu数量决定了训练时的batch_size。如下图, 由于我只有一个gpu, 该参数设置为 2, 所以 batch_size为2

- 学习率设置: 默认学习速率为8个gpu。如果使用的GPU小于或大于8个，则需要设置学习速率与GPU个数成正比，例如4个GPU的学习速率为0.01,16个GPU的学习速率为0.04。
同理 1 个 GPU, samples_per_gpu= 2, 学习速率设置为 0.0025
计算公式： 批大小(gpu_num * samples_per_gpu) / 16 * 0.02

(4) 使用预训练模型训练
需要提前下载预训练模型, 可以从如下链接中获取需要的模型: https://github.com/open-mmlab/mmdetection/blob/master/docs_zh-CN/model_zoo.md
然后修改模型配置文件 dfft_time.py 中的
load_from = None
# 修改为:
load_from = '/home/mayi/wd/lmy/ObjectDetection/DFFT/dfft_medium_3x_2.pth'
# 其中具体目录要看放在哪个目录下

(5) load_from, resume_from, pre_train 的区别
- resume_from 同时加载模型权重(model weights) 和优化状态(optimizer status)，且 epoch 是继承了指定 checkpoint 的信息. 一般用于意外终端的训练过程的恢复.
- load_from 仅加载模型权重(model weights)，训练过程的 epoch 是从 0 开始训练的, 相当于重新开始, 一般用于模型 finetuning(微调).
- 如果要使用, 两者都为’/work_dir/xxx/epoch_xxx.pth’ 格式
- 加载顺序优先级: pretrained, resume_from> load_from , 其中如果加载了 resume_from的 断点文件, 那么久不会再加载 load_from 的文件
```

6. 训练
```
# 运行命令开始训练, 注意这里使用刚刚修改的在 work_dir(如our_pre) 下生成的 dfft.py 文件进行训练：
python tools/train.py our_coco_time/dfft_time.py

# 后台运行，写入日志
nohup python tools/train.py our_coco_time/dfft_time.py --work-dir our_coco_time  > log_file/our.log 2>&1 &

```

```angular2html
最优权重和日志：
our数据集：./our_coco_time/resume/resume/epoch_52.pth，/home/mayi/wd/lmy/ObjectDetection/Improved_DFFT/our_coco_time/20230720_203845.log.json
和/home/mayi/wd/lmy/ObjectDetection/Improved_DFFT/our_coco_time/resume/resume/20230725_225252.log.json

kaggle数据集：./kaggle_temp/pre_new/resume/epoch_236_8300.pth，日志：/home/mayi/wd/lmy/ObjectDetection/Improved_DFFT/kaggle_temp/pre_new/resume/20230824_203034.log.json
```

7. 测试命令
```angular2html
# 改进时修改了mmdet/backbones/DFFTNet_time.py文件，对不同数据集测试时需要改变该文件中的模板路径。
# 改完后运行python setup.py install，然后运行下列命令测试
python tools/test.py ./our_coco_time/resume/dfft_time.py ./our_coco_time/resume/resume/epoch_52.pth --eval-options  'classwise=True' 'iou_thrs=[0.5]' --eval bbox
python tools/test.py ./kaggle_temp/pre_new/resume/dfft.py ./kaggle_temp/pre_new/resume/epoch_236_8300.pth --eval-options  'classwise=True' 'iou_thrs=[0.5]' --eval bbox
```

8. 推理
```angular2html
python predict_our.py
```

9. 记录fps
```
# 找到 tools/analysis_tools/benchmark.py，修改关键参数
python tools/analysis_tools/benchmark.py ./our_coco_time/resume/dfft_time.py ./our_coco_time/resume/resume/epoch_52.pth
python tools/analysis_tools/benchmark.py ./kaggle_temp/pre_new/resume/dfft.py ./kaggle_temp/pre_new/resume/epoch_236_8300.pth
```