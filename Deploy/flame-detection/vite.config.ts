import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { viteMockServe } from 'vite-plugin-mock'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    viteMockServe({
      mockPath: 'mock',
      enable: true,
      logger: true,
    }),
  ],
  server: {
    proxy: {
      // 视频流服务：CameraFeed_flask
      '/camera/': {
        // target: 'http://localhost:5002',
        target: 'http://10.88.76.75:5002',
        changeOrigin: true,
      },

      // 分类模型Classification_flask
      // 添加Socket.io代理配置
      '/socket.io': {
        // target: 'http://localhost:5001',
        target: 'http://10.88.76.75:5001',
        changeOrigin: true,
        ws: true, // 启用WebSocket代理
      },

      // 前端无需关注目标检测模型Classification_flask服务，在Classification_flask中才会调用Classification_flask
    }
  },
})
