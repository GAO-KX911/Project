# 注意：以下各步骤中的主机操作和docker操作可自主进行选择（二选一）；运行文件时根据自己需要选择启动联调页面还是启动分类页面
1. 激活虚拟环境
主机操作：conda activate class_zhang
docker操作：conda activate zzl_classification

2. 进入文件目录
主机操作：cd /xxx/Classification_flask
docker操作：cd /app/fire-code/Classification_flask

3. 运行文件
联调操作：python app-textIN-textOUT-memory.py
分类操作：python app-textIN-textOUT.py
