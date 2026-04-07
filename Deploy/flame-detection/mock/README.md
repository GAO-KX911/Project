# Mock 数据配置说明

## 简介

本项目使用 `vite-plugin-mock` 插件来模拟后端 API 数据，无需搭建真实的后端服务器即可进行前端开发。

## 配置说明

已在 `vite.config.ts` 中配置了 `vite-plugin-mock` 插件：

```typescript
import { viteMockServe } from 'vite-plugin-mock'

export default defineConfig({
  plugins: [
    react(),
    viteMockServe({
      mockPath: 'mock', // mock文件所在目录
      localEnabled: true, // 开发环境启用
      prodEnabled: false, // 生产环境禁用
      injectCode: false, // 不注入到页面
      logger: true, // 在控制台显示请求日志
    }),
  ],
})
```

## 使用方法

### 1. 创建 Mock 数据

所有的 mock 数据文件都放在 `mock` 目录下，默认入口文件是 `mock/index.ts`。

每个 mock 接口需要定义：
- `url`: 接口地址
- `method`: 请求方法（get, post, put, delete等）
- `response`: 响应函数，返回模拟数据

### 2. 示例

已创建了几个示例接口：

```typescript
// 获取用户列表
{
  url: '/api/users',
  method: 'get',
  response: () => {
    // 返回模拟数据
    return {
      code: 200,
      data: [...],
      message: '获取用户列表成功'
    }
  }
}
```

### 3. 在组件中使用

在 React 组件中，直接使用 fetch 或 axios 等工具请求这些 mock 接口即可：

```typescript
// 示例：获取用户列表
fetch('/api/users')
  .then(response => response.json())
  .then(result => {
    if (result.code === 200) {
      // 处理数据
      console.log(result.data)
    }
  })
```

## 高级用法

### 使用 MockJS 生成随机数据

本项目已集成 `mockjs` 库，可以用来生成更真实的随机数据：

```typescript
import Mock from 'mockjs'
const Random = Mock.Random

// 示例：生成随机用户数据
const user = {
  id: Random.id(),
  name: Random.cname(),
  email: Random.email(),
  avatar: Random.image('100x100')
}
```

### 模拟请求延迟

可以使用 setTimeout 模拟网络延迟：

```typescript
response: () => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        code: 200,
        data: { /* 数据 */ },
        message: '成功'
      })
    }, 1000) // 延迟1秒
  })
}
```

