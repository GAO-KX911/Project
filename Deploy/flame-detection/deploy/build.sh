cd ../
yarn build

# 构建 AMD64 平台（默认）
docker build --platform linux/amd64 -f deploy/Dockerfile -t flame-client:amd64 .
# 构建 ARM64 平台
docker build --platform linux/arm64 -f deploy/Dockerfile -t flame-client:arm64 .

# 导出镜像（选择需要的平台）
docker save -o deploy/client-docker/flame-client-amd64.tar flame-client:amd64
docker save -o deploy/client-docker/flame-client-arm64.tar flame-client:arm64

