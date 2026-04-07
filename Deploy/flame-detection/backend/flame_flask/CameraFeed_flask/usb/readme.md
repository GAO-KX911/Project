# 通用USB摄像头视频流
## 1. 部署
### 1.0 安装v4l2-ctl
为了一套代码兼容amd和arm平台摄像头，使用v4l2-ctl来检测设备类型，所有还需要安装v4l2-ctl
```bash
# 检查是否已安装
v4l2-ctl --version
# 没有则需要安装
sudo apt update
sudo apt install v4l-utils
```

### 1.1 Jetson平台，IMX219摄像头
这个平台系统貌似自带opencv, 如果新建conda env再安装opencv反而会冲突，不如直接使用系统的python中安装依赖和运行。
可以使用下面命令验证
```bash
# 查看系统 Python 路径
which python3

# 查看系统 OpenCV (cv2) 路径
python3 -c "import cv2; print(cv2.__file__)"
```
如果系统已安装 OpenCV，通常会输出类似：
```bash
/usr/bin/python3
/usr/lib/python3/dist-packages/cv2.cpython-38-aarch64-linux-gnu.so
```

安装依赖
```bash
pip install Flask==2.2.5 gunicorn==20.1.0
```

### 1.2 AMD平台
非Jetson平台，一般机器应该是这种
```bash
# 创建conda环境
conda env create -f environment.yml
conda activate lzn_camera

# 如果是微型机的Jetson平台, 自带opencv， 貌似不能额外安装，视频流会冲突没法显示； 其它平台需要安装
conda install -c conda-forge opencv
```
### 1.3 后台运行
```bash
nohup gunicorn -w 1 -b 0.0.0.0:5002 app:app &
```


## 2. 微型机备注
* OS: Ubuntu "20.04.6 LTS (Focal Fossa)"
* CPU架构: arm64 (aarch64)
* Jetson平台，IMX219摄像头
### 查看摄像头
```bash
# 列举设备
v4l2-ctl --list-devices
```
### 使用命令截图并保存
可测试摄像头是否可用
```bash
gst-launch-1.0 nvarguscamerasrc num-buffers=1 ! \
'video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1' ! \
nvvidconv ! jpegenc ! filesink location=capture.jpg
```
### 截图能成功，视频流失败
可能是opencv冲突，conda环境中无需额外安装opencv, 得使用Jetson平台自带的opencv
```bash
pip uninstall opencv-python opencv-python-headless
# 这一句能输出成功，基本就可以了
python3 -c "import cv2; print(cv2.getBuildInformation())"
```

