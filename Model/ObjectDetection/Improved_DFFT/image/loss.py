import json
import matplotlib.pyplot as plt

# 读取 JSON 日志文件并提取数据
with open('../new_loss.json', 'r') as f:
    logs = [json.loads(line.strip()) for line in f]

epochs = []
loss_cls = []
loss_bbox = []

for log in logs:
    if 'mode' in log and log['mode'] == 'train':
        epochs.append(log['epoch'])
        loss_cls.append(log['loss_cls'])
        loss_bbox.append(log['loss_bbox'])

# 绘制曲线图
plt.plot(epochs, loss_cls, linestyle='-', label='loss_cls')
plt.plot(epochs, loss_bbox, linestyle='-', label='loss_bbox')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()

# 保存曲线图为图片
plt.savefig('our_loss_bbox_and_cls.png')
plt.show()