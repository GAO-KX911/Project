#!/bin/bash
PORTS=(5001 7866 5002)

echo "1. 停止后端三个服务, 🔫 杀死端口 5001, 7866, 5002 的进程..."

for PORT in "${PORTS[@]}"; do
    echo "🔍 检查端口 $PORT ..."
    
    # 使用更精确的方法提取PID
    PIDS=$(ss -tunlp | grep ":$PORT" | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u)
    
    if [ -z "$PIDS" ]; then
        echo "✅ 端口 $PORT 空闲"
        continue
    fi
    
    echo "📋 找到进程: $PIDS"
    
    # 杀死进程
    for PID in $PIDS; do
        # 检查进程是否存在
        if ps -p $PID > /dev/null 2>&1; then
            echo "🛑 杀死进程 PID: $PID (端口 $PORT)"
            kill -9 $PID
            if [ $? -eq 0 ]; then
                echo "✅ 进程 $PID 已杀死"
            else
                echo "❌ 无法杀死进程 $PID"
            fi
        else
            echo "⚠️  进程 $PID 不存在"
        fi
    done
done

echo -e "🎯 后端服务已停止！\n"

echo "2. 停止前端docker容器"
docker stop fire-client
echo "前端docker容器停止"
