
# Flame Detection
本项目是一个基于`vite + react`开发的 火焰/烟雾检测系统前端应用，支持远程摄像头、本地摄像头、本地视频的检测。前后端采用`WebSocket`实时通信。

## 目录结构
```
flame-detection/
├── backend/
│   └── flame_flask/                           # 后端服务部分文件
├── deploy/
│   └── default.conf                           # 前端生产环境部署文件
├── src/
│   ├── components/                            # React 组件
│   │   └── useInteval.ts
│   └── App.tsx                                # 前端主入口
├── package.json                               # 前端依赖
├── vite.config.ts                             # Vite 配置
└── README.md
```

## 快速开始
### 1. 安装前端依赖
```sh
yarn
```

### 2. 启动本地开发服务器
```sh
yarn dev
```

### 3. 本地开发时对接后端服务
#### 方式1：对接真实后端服务
vite.config.ts修正target为服务器地址

### 方式2：本地mock后端服务 （可忽略）
```bash
# 1. 修改vite.config.ts的target为localhost

# 2. 启动mock服务
sh backend/run.sh
```

### 4. 打包docker镜像
```bash
cd deploy
sh build.sh
```
生成的docker镜像会放在`deploy/client-docker`目录下

### 5. 手机端访问
已经做了手机端的自适应。
但是ip地址类型的url，微信好像会拦截，在手机端浏览器中可以正常访问

### 6. 服务器部署
查看 `deploy/client-docker/readme.md`