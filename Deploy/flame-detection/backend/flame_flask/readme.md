最后可以使用gunicorn + nohup 启动

# 1. 识别服务
```bash
cd Classification_flask
conda activate lzn_fc # class_zhang
# Gunicorn（Green Unicorn）是一个 Python WSGI(Web Server Gateway Interface) HTTP Server，更稳定高效
# -k gevent: 使用gevent异步worker（适合 I/O 密集型应用）
# -w: 多少个worker，由于socket特殊性，此处设置1通信结果更加一致
nohup gunicorn -k gevent -w 1 -b 0.0.0.0:5001 app-textIN-textOUT-memory:app &
```

# 2. 检测服务
```bash
conda activate lzn_ob  # OB_ZHANG

# 这一步作用是干吗呢？
cd /home/mayi/wd/zzl/ObjectDetection/Improved_DFFT
# cd /home/mayi/wd/ObjectDetection/Improved_DFFT
python setup.py install

cd ObjectDetection_flask  
nohup gunicorn -b 0.0.0.0:7866 app-textIN-textOUT-memory:app &
```

# 3. 视频流
```bash
cd CameraFeed_flask

# 1. 微型机中不使用conda环境, 使用系统python
conda deactivate 
# 2. 非微型机可以使用conda环境
conda activate lzn_camera
nohup gunicorn -b 0.0.0.0:5002 hk_camera:app &
```

# 如果后端也用容器启动
```bash
sudo docker run -d -v /home/mayi/wd/shu-njust:/app/fire-code -p 5001:5001 --gpus all --name fire-model-3.0 --add-host=host.docker.internal:172.17.0.1 my_project:v3.0 bash /app/fire-code/start.sh
```
