# Shuffle_PSA方法

### 文件及文件夹介绍

- /Our  存放对于自建数据集Our_flame_smoke的训练好的最优模型权重和指标记录文件（train_val_metrics.txt）

- /FLAME  存放对于FLAME数据集的训练好的最优模型权重和指标记录文件（train_val_metrics.txt）

- /DeepQuestAI  存放对于DeepQuestAI数据集的训练好的最优模型权重和指标记录文件（train_val_metrics.txt）

- /log_file 存放训练时的日志文件

- /Predict 在Our_flame_smoke和FLAME两个数据集上的推理结果，用于找出案例分析中的例子

- /Resolution_test 分辨率分析中的文件
```
- img_resolution.py # 改变原始图片的分辨率
- eval_resolution.py # 测试模型在不同分辨率下的性能
- eval.sh # 批量测试性能
```

- /Ablation_study 不添加psa的消融研究
```
- /Our  存放对于自建数据集Our_flame_smoke的训练好的最优模型权重和指标记录文件（train_val_metrics.txt）
- /FLAME  存放对于FLAME数据集的训练好的最优模型权重和指标记录文件（train_val_metrics.txt）
- /DeepQuestAI  存放对于DeepQuestAI数据集的训练好的最优模型权重和指标记录文件（train_val_metrics.txt）

- train_our.py 文件为自建数据集Our_flame_smoke的训练文件
- train_flame.py 文件为FLAME数据集的训练文件
- train_deep.py 文件为DeepQuestAI数据集的训练文件

- eval_our.py 文件为自建数据集Our_flame_smoke的测试文件
- eval_flame.py 文件为FLAME数据集的测试文件
- eval_deep.py 文件为DeepQuestAI数据集的测试文件
```

- CAM_Attention 注意力可视化分析
```
- /Our  存放对于自建数据集Our_flame_smoke的注意力分析结果，其中./Input下为输入图片，./Output下为可视化结果
- /FLAME  存放对于FLAME数据集的注意力分析结果，其中./Input下为输入图片，./Output下为可视化结果
- /DeepQuestAI  存放对于DeepQuestAI数据集的注意力分析结果，其中./Input下为输入图片，./Output下为可视化结果

- grad_cam.py # 注意力可视化展示
- utils.py # 注意力可视化展示过程中所需要的依赖方法
```

- /models 下为Shuffle_PSA模型的网络结构定义文件
```
- shuffle_psa.py # Shuffle_PSA模型的网络结构定义文件
- shuffle_wopsa.py # 不添加PSA模块的模型的网络结构定义文件
- loss.py # 加权的交叉熵损失定义文件
```

- train_our.py 文件为自建数据集Our_flame_smoke的训练文件

- train_flame.py 文件为FLAME数据集的训练文件

- train_deep.py 文件为DeepQuestAI数据集的训练文件

- eval_our.py 文件为自建数据集Our_flame_smoke的测试文件

- eval_flame.py 文件为FLAME数据集的测试文件

- eval_deep.py 文件为DeepQuestAI数据集的测试文件

- count_param_fps.py 文件为测试模型参数量和fps的文件

- shufflenetv2_x1.pth  预训练的权重文件

- predict.py  批量火灾图片的推理文件

### 训练、测试和推理流程
1. 激活环境
```
conda activate fire_class
```
2. 进入当前目录，运行各类train_*.py文件进行模型训练
```
python train_our.py
# 在Our_flame_smoke数据集上训练模型并进行测试
```

```
nohup python train_our.py  > ./log_file/our.log 2>&1 &
# 后台运行并保存日志
```

3. 运行各类eval_*.py文件和count_param_fps.py文件进行模型性能测试
```
python eval_our.py
# 在Our_flame_smoke数据集上进行测试
python count_param_fps.py  
# 修改该文件中的自定义参数计算参数量和fps
```

4. 运行predict.py进行批量火灾图片推理识别，找出案例
```
python predict.py
```