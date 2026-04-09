#!/bin/bash

# 加载conda
source /home/njust/Conda/etc/profile.d/conda.sh

# 统一读取部署配置
DEPLOY_ENV_FILE="/home/njust/Fire/Deploy/deploy.env"
if [ -f "$DEPLOY_ENV_FILE" ]; then
    set -a
    . "$DEPLOY_ENV_FILE"
    set +a
fi

echo "🚀 一键启动所有服务..."
echo ""
echo "🧩 当前部署环境: ${DEPLOY_ENV_NAME:-default}"
echo ""

# 函数：检查端口是否被占用
check_port() {
    local port=$1
    if ss -tunlp | grep -q ":$port "; then
        return 0  # 端口被占用
    else
        return 1  # 端口空闲
    fi
}

# 函数：启动服务（带端口检测）
start_service() {
    local service_name=$1
    local port=$2
    local start_command=$3
    
    echo -n "$service_name (端口:$port)... "
    
    if check_port $port; then
        echo "✅ 已启动（跳过）"
        return 0
    else
        eval $start_command
        # 等待5秒让服务启动
        sleep 5
        
        if check_port $port; then
            echo "✅ 启动成功"
            return 0
        else
            echo "❌ 启动失败"
            return 1
        fi
    fi
}

# 显示端口状态
echo "🔍 检查端口状态..."
ports=("5001:分类模型" "7866:检测模型" "5002:相机服务" "80:前端服务" "443:前端HTTPS")
for port_info in "${ports[@]}"; do
    IFS=':' read -r port service <<< "$port_info"
    if check_port $port; then
        echo "  ✅ 端口 $port ($service) - 运行中"
    else
        echo "  ❌ 端口 $port ($service) - 未启动"
    fi
done

echo ""
echo "🎯 开始启动服务..."

# 启动分类模型
echo "1. 🔥 启动分类模型"
start_service "分类模型" "5001" "
    conda activate classification && 
    cd /home/njust/Fire/Deploy/Classification_flask && 
    nohup gunicorn -k gevent -w 1 -b 0.0.0.0:5001 app-textIN-textOUT-memory:app > classification.log 2>&1 &
"

# 启动检测模型
echo "2. 🔍 启动检测模型"
start_service "检测模型" "7866" "
    conda activate dfft && 
    cd /home/njust/Fire/Deploy/ObjectDetection_flask && 
    nohup gunicorn -b 0.0.0.0:7866 app-textIN-textOUT-memory:app > detection.log 2>&1 &
"

# 启动相机服务
echo "3. 📷 启动相机服务"
start_service "相机服务" "5002" "
    conda activate camera && 
    cd /home/njust/Fire/Deploy/CameraFeed_flask && 
    nohup python hk_camera.py > camera.log 2>&1 &
"

# 启动前端docker容器
echo "4. 🌐 启动前端监控服务"
echo -n "前端服务 (端口:80,443)... "

# 检查docker容器状态
if docker ps | grep -q "fire-client"; then
    echo "✅ 已运行（跳过）"
else
    docker start fire-client > /dev/null 2>&1
    sleep 3
    
    if docker ps | grep -q "fire-client"; then
        echo "✅ 启动成功"
    else
        echo "❌ 启动失败"
    fi
fi

echo ""
echo "🎉 服务启动完成！"
echo ""
echo "📊 最终状态检查:"

# 最终状态验证
services=(
    "5001:分类模型"
    "7866:检测模型" 
    "5002:相机服务"
    "80:前端HTTP"
    "443:前端HTTPS"
)

all_running=true
for service in "${services[@]}"; do
    IFS=':' read -r port name <<< "$service"
    if check_port $port; then
        echo "  ✅ $name (端口:$port) - 运行中"
    else
        echo "  ❌ $name (端口:$port) - 未运行"
        all_running=false
    fi
done

echo ""
echo "🌐 服务访问地址:"
echo "  🔹 前端界面: http://127.0.0.1 (或你的服务器IP)"
echo "  🔹 分类模型: http://127.0.0.1:5001"
echo "  🔹 检测模型: http://127.0.0.1:7866"
echo "  🔹 相机服务: http://127.0.0.1:5002"

if $all_running; then
    echo ""
    echo "🎊 所有服务正常运行！"
else
    echo ""
    echo "⚠️  部分服务启动异常，请检查日志文件"
    echo "📝 查看日志:"
    echo "  tail -f /home/njust/Fire/Deploy/Classification_flask/classification.log"
    echo "  tail -f /home/njust/Fire/Deploy/ObjectDetection_flask/detection.log" 
    echo "  tail -f /home/njust/Fire/Deploy/CameraFeed_flask/camera.log"
fi
