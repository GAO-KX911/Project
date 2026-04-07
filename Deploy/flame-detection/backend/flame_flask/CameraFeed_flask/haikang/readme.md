# 海康相机启动
## 1. 安装海康相机自己的驱动

## 2. 启动服务
```bash
conda activate lzn_camera
cd /home/njust/Fire/Deploy/CameraFeed_flask
nohup gunicorn -b 0.0.0.0:5002 hk_camera:app &
```