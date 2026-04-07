import json
import matplotlib.pyplot as plt

# 读取 JSON 日志文件并提取数据
with open('new_loss.json', 'r') as f:
    logs = [json.loads(line.strip()) for line in f]

epochs = []
bbox_mAP = []

for log in logs:
    if 'mode' in log and log['mode'] == 'val':
        epochs.append(log['epoch'])
        bbox_mAP.append(log['bbox_mAP'])

# 绘制曲线图
plt.plot(epochs, bbox_mAP, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('bbox_mAP')
plt.title('bbox_mAP')
plt.grid(False)
plt.savefig('./image/bbox_mAP.png')
plt.show()