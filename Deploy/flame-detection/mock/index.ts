import { MockMethod } from 'vite-plugin-mock'
import Mock from 'mockjs'

// 使用mockjs生成随机数据
const Random = Mock.Random

export default [
  {
    url: '/upload',
    method: 'post',
    response: () => {
      return {
        "detections": [
          // 分类种类
          {
            "class": "smoke",
            "confidence": 0.48,
            // 坐标
            "coordinates": [
              // 矩形左上坐标
              [
                552.48,
                99.47
              ],
              // 矩形右下角
              [
                639.19,
                359.24
              ]
            ]
          },
          {
            "class": "fire",
            "confidence": 0.9,
            // 坐标
            "coordinates": [
              // 矩形左上坐标
              [
                Random.integer(10, 100),
                Random.integer(120, 160),
              ],
              // 矩形右下角
              [
                Random.integer(110, 150),
                Random.integer(200, 250),
              ]
            ]
          }
        ],
        // 图片的尺寸
        "dimensions": {
          "height": 360,
          "width": 640
        },
        "time": "61.040"
      }
    }
  },

  {
    url: '/predict',
    method: 'post',
    response: () => {
      return {
        "inference_time": 25747.294,
        "result": [
          "分类结果：火焰               概率：0.934",
          "分类结果：烟雾               概率：0.034",
          "分类结果：中性               概率：0.032"
        ]
      }
    }
  },

  {
    url: '/predict_path',
    method: 'post',
    response: () => {
      return {
        "inference_time": 2.12,
        "result": [
          "分类结果：火焰               概率：0.934",
          "分类结果：烟雾               概率：0.034",
          "分类结果：中性               概率：0.032"
        ]
      }
    }
  },

  {
    url: '/detect_by_path',
    method: 'post',
    response: () => {
      return {
        "detections": [
          {
            "class": "fire",
            "confidence": 0.47,
            "coordinates": [
              [
                1223.69,
                759.98
              ],
              [
                1477.57,
                1068.73
              ]
            ]
          },
          {
            "class": "fire",
            "confidence": 0.43,
            "coordinates": [
              [
                913.36,
                614.1
              ],
              [
                1173.23,
                1070.0
              ]
            ]
          },
          {
            "class": "fire",
            "confidence": 0.28,
            "coordinates": [
              [
                1216.86,
                808.49
              ],
              [
                1356.37,
                1041.32
              ]
            ]
          },
          {
            "class": "fire",
            "confidence": 0.25,
            "coordinates": [
              [
                121.98,
                938.51
              ],
              [
                207.83,
                1030.11
              ]
            ]
          },
          {
            "class": "fire",
            "confidence": 0.25,
            "coordinates": [
              [
                1357.02,
                770.47
              ],
              [
                1478.0,
                1063.54
              ]
            ]
          }
        ],
        "dimensions": {
          "height": 1070,
          "width": 1478
        },
        "time": "279.153"
      }
    }
  }


] as MockMethod[]