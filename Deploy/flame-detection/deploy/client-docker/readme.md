# 使用docker镜像进行前端部署
## 1. 安装docker
检查环境是否已经安装了docker
```bash
docker -v
```
如果没有安装则使用以下命令安装
```bash
# 1. 安装docker
sudo apt update
sudo apt install docker.io

# 2. 启动并设置Docker开机自启
sudo systemctl start docker
sudo systemctl enable docker
```

## 2. 部署前端服务
### 2.1 导入docker镜像
* 如果系统是amd架构，使用flame-client-amd64.tar
* 如果系统是arm架构，使用flame-client-arm64.tar
```bash
# 结果为x86_64是AMD架构；如果是aarch64则ARM架构
uname -m
```

把flame-client-amd64.tar上传到服务器后
```bash
# 加载镜像
sudo docker load -i flame-client-amd64.tar
# 查看镜像
sudo docker images
```
## 2.2 启动服务
```bash
# -p: publish，容器默认隔离网络，手动映射端口才能从外部访问服务, 在容器内部启动了http的80和https的443端口，如果服务器443端口被占用了，可以改为其他端口，例如：-p 8080:80 -p 8443:443
# -d: detach, 分离模式，后台运行
# --add-host: docker容器内调用宿主机服务。docker会虚拟一个网关，默认是172.17.0.1；可以使用 `ip a show docker0` 确认
sudo docker run -d -p 80:80 -p 443:443 --add-host=host.docker.internal:172.17.0.1 --name fire-client flame-client:amd64
```

