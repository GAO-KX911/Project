# 使用官方nginx镜像作为基础镜像
FROM nginx:alpine

# 安装openssl用于生成自签名证书, 不缓存包索引（减少镜像体积）
RUN apk add --no-cache openssl

# 创建SSL证书目录
RUN mkdir -p /etc/nginx/ssl

# 生成自签名SSL证书
RUN openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/nginx/ssl/self.key \
    -out /etc/nginx/ssl/self.crt \
    -subj "/C=CN/ST=State/L=City/O=Organization/CN=localhost"

# 复制前端构建文件到nginx默认目录（从根目录的相对路径）
COPY dist /usr/share/nginx/html

# 复制nginx配置文件
COPY deploy/default.conf /etc/nginx/conf.d/default.conf

# 设置正确的权限
RUN chown -R nginx:nginx /usr/share/nginx/html \
    && chown -R nginx:nginx /etc/nginx/ssl

# 暴露端口
EXPOSE 80 443

# 启动nginx
CMD ["nginx", "-g", "daemon off;"]