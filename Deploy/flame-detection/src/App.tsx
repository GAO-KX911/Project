import { useEffect, useRef, useState } from 'react';
import { Card, message, Spin } from 'antd';
import videoSource from './assets/10.mp4';
import { io, Socket } from 'socket.io-client';
import RoiArea from './components/RoiArea';
import { useInterval } from './components/useInteval';
import { useLocalCamera } from './components/useLocalCamera';
import Controller from './components/Controller';
import Predict from './components/Predict';
import { VideoCameraOutlined } from '@ant-design/icons';


export default function App() {
  // 左上角按钮
  const [isDetecting, setIsDetecting] = useState(false);  // 添加检测状态
  const [isRoiMode, setIsRoiMode] = useState<boolean>(false); // roi区域
  // 添加 ROI区域
  const [roiRect, setRoiRect] = useState<RectType>({ x: 0, y: 0, width: 0, height: 0 });
  // 监控视频类型：远程摄像头、本地摄像头、样本视频、更换视频等
  const [monitorType, setMonitorType] = useState<MonitorType>(localStorage.getItem("monitorType") as MonitorType || 'remote_camera');
  const monitorTypeRef = useRef<MonitorType>(monitorType);

  const clearCanvas = (canvas: HTMLCanvasElement | null) => {
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
    }
  };

  const [remoteCameraLoading, setRemoteCameraLoading] = useState<boolean>(true)

  // 2个定时任务
  const apiInvokeIntervalRef = useRef<string>(localStorage.getItem("apiInvokeInterval") || "600");
  const classifyThresholdRef = useRef<string>(localStorage.getItem("classifyThreshold") || "0.6");
  const detectThresholdRef = useRef<string>(localStorage.getItem("detectThreshold") || "0.6")

  // 添加WebSocket连接状态
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isSocketConnected, setIsSocketConnected] = useState(false);

  // 引用界面上的dom元素
  const imgDomRef = useRef<HTMLImageElement>(null);
  const { videoDomRef } = useLocalCamera(monitorType === 'local_camera');
  const monitorDomRef = useRef<HTMLImageElement | HTMLVideoElement>(null);
  useEffect(() => {
    if (monitorType === 'remote_camera') {
      monitorDomRef.current = imgDomRef.current;
    } else {
      monitorDomRef.current = videoDomRef.current;
    }
  }, [monitorType, remoteCameraLoading])

  // 切换视频时清空检测结果
  useEffect(() => {
    monitorTypeRef.current = monitorType
    // 清空结果区域
    setPredictResult(undefined)
    setRoiPredictResult(undefined)
    clearCanvas(detectionCanvasRef.current)
    clearCanvas(roiDetectionCanvasRef.current)
  }, [monitorType, videoDomRef?.current?.src]);

  // 用于截图，然后往后端传输图片
  const monitorCanvasRef = useRef<HTMLCanvasElement>(document.createElement('canvas'));

  // 检测结果
  const detectionCanvasRef = useRef<HTMLCanvasElement>(null);
  // roi区域绘图
  const roiCanvasRef = useRef<HTMLCanvasElement>(null);
  // ROI检测结果
  const roiDetectionCanvasRef = useRef<HTMLCanvasElement>(null);

  // 识别结果
  const [predictResult, setPredictResult] = useState<PredictType>();
  const [roiPredictResult, setRoiPredictResult] = useState<PredictType>();
  // 检测时间
  const [detectTime, setDetectTime] = useState<string>();
  const [roiDetectTime, setRoiDetectTime] = useState<string>();

  // 初始化WebSocket连接
  useEffect(() => {
    const newSocket = io();

    newSocket.on('connect', () => {
      // 不是初次连接，应该是断了再重连的情况，给个界面提示
      if (socket) {
        message.success('WebSocket连接成功');
      }

      setIsSocketConnected(true);
    });

    newSocket.on('disconnect', () => {
      message.error('WebSocket连接断开');
      setIsSocketConnected(false);
    });

    // 监听上传响应
    newSocket.on('upload_image_result', (data) => {
      console.log('上传成功:', data.message);
    });

    newSocket.on('predict_result', (data: PredictType) => {
      const { prediction_type, error } = data;

      if (error) {
        console.log("识别结果失败", prediction_type, error);
        return;
      }

      if (prediction_type === "original") {
        setPredictResult(data);
      } else {
        setRoiPredictResult(data);
      }
      console.log(`识别成功 ${prediction_type}:`, data);
    });

    newSocket.on('detect_result', (data: DetectionType) => {
      const { detection_type, time, error } = data;

      if (error) {
        console.log("检测结果失败", detection_type, error);
        return;
      }

      if (!detectionCanvasRef.current || !roiDetectionCanvasRef.current || !monitorDomRef.current || !roiCanvasRef.current) {
        return
      }

      if (detection_type === "original") {
        setDetectTime(time);
        drawRedRect(data, detectionCanvasRef.current, monitorDomRef.current);
      } else {
        setRoiDetectTime(time);
        drawRedRect(data, roiDetectionCanvasRef.current, roiCanvasRef.current);
      }

      console.log(`检测成功: ${detection_type}`, data);
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, []);

  useEffect(() => {
    window.addEventListener("resize", initRoiRect);
    return () => {
      window.removeEventListener("resize", initRoiRect);
    }
  }, []);

  const initRoiRect = () => {
    if (!monitorDomRef.current) return;

    // 小屏幕下随便设个roiRect大小可能会超过视频区域，所以需要优化限制一下
    const monitorRect = monitorDomRef.current.getBoundingClientRect();

    setRoiRect({
      x: 80,
      y: 50,
      width: monitorRect.width * 0.3,
      height: monitorRect.height * 0.5
    })
  }

  const drawRedRect = (data: DetectionType, canvas: HTMLCanvasElement, source: CanvasImageSource) => {
    // 在canvas上绘制检测结果
    // 设置canvas尺寸与图片一致，如果后端没有改变图片大小，则不需要设置
    canvas.width = data.dimensions.width;
    canvas.height = data.dimensions.height;
    const ctx = canvas.getContext('2d');
    // 原图绘制
    ctx?.drawImage(source, 0, 0, canvas.width, canvas.height)

    if (ctx) {
      // 计算缩放比例
      const canvasRect = canvas.getBoundingClientRect();
      const scaleX = canvasRect.width / canvas.width;
      const scaleY = canvasRect.height / canvas.height;
      // 使用(scaleX, scaleY)较小的缩放比例保持宽高比, 
      // 1是roi显示区域可能会被真实图片更大，例如14/2, 反而是在压缩了
      const scale = Math.min(scaleX, scaleY, 1);

      // 根据缩放比例调整文字大小和背景尺寸
      const adjustedFontSize = 14 / scale; // 反向缩放文字大小
      const adjustedBackgroundHeight = 20 / scale;
      const adjustedLineWidth = 3 / scale;

      // 为每个检测结果绘制矩形框
      data.detections.forEach(detection => {
        if (detection.confidence >= parseFloat(detectThresholdRef.current) && detection.coordinates && detection.coordinates.length === 2) {

          const [topLeft, bottomRight] = detection.coordinates;
          const width = bottomRight[0] - topLeft[0];
          const height = bottomRight[1] - topLeft[1];

          // smoke使用蓝色吧
          // 设置矩形样式
          ctx.strokeStyle = detection.class === 'smoke' ? 'blue' : 'red';
          ctx.lineWidth = adjustedLineWidth;
          ctx.strokeRect(topLeft[0], topLeft[1], width, height);

          // 添加标签
          ctx.fillStyle = detection.class === 'smoke' ? 'rgba(0, 0, 255, 0.6)' : 'rgba(255, 0, 0, 0.6)';
          ctx.fillRect(topLeft[0], topLeft[1], width, adjustedBackgroundHeight);
          if (detection.class && detection.confidence) {
            ctx.fillStyle = 'white';
            ctx.font = `${adjustedFontSize}px Arial`;
            ctx.textBaseline = 'middle'; // 设置文字基线为中间
            ctx.fillText(`${detection.class}: ${(detection.confidence * 100).toFixed(2)}%`, topLeft[0] + 5 / scale, topLeft[1] + adjustedBackgroundHeight / 2, width - 5 / scale);
            ctx.textBaseline = 'alphabetic'; // 恢复默认基线
          }
        }
      });
    }
  };

  const invokeApi = async () => {
    if (!socket || !isSocketConnected) {
      console.error('WebSocket未连接');
      return
    }

    let base64Data = "";
    if (monitorTypeRef.current !== "remote_camera") {
      if (!videoDomRef.current || videoDomRef.current.paused) return;

      // 截取完整帧
      monitorCanvasRef.current.getContext('2d')?.drawImage(videoDomRef.current, 0, 0);
      // 转换为base64格式发送, jpg是不是比png压缩厉害，体积小，然后传输速度快？
      // 添加质量参数（0-1之间的值，1表示最高质量）
      base64Data = monitorCanvasRef.current.toDataURL('image/jpeg', 0.92);
    }

    socket.emit('upload_image', {
      monitorType: monitorTypeRef.current,
      classifyThreshold: classifyThresholdRef.current,
      // remote_camera类型，不用发送图像，后端可以自行获取
      image: base64Data,
      roiRect: isRoiMode ? transRectToRealSize(roiRect) : null
    });
  }

  const transRectToRealSize = (rect: RectType): RectType => {
    if (!monitorDomRef.current) {
      return rect;
    }

    // 使用 getBoundingClientRect 获取实际渲染尺寸
    const videoRect = monitorDomRef.current.getBoundingClientRect();
    let realWidth = 0
    let realHeight = 0
    if (monitorTypeRef.current === 'remote_camera' && imgDomRef.current) {
      realWidth = imgDomRef.current.naturalWidth
      realHeight = imgDomRef.current.naturalHeight
    } else if (videoDomRef.current) {
      realWidth = videoDomRef.current.videoWidth
      realHeight = videoDomRef.current.videoHeight
    }
    // 计算比例，使用视频的实际尺寸
    const widthRatio = realWidth / (videoRect.width || 1);
    const heightRatio = realHeight / (videoRect.height || 1);

    return {
      x: rect.x * widthRatio,
      y: rect.y * heightRatio,
      width: rect.width * widthRatio,
      height: rect.height * heightRatio
    };
  }

  // 2个定时器  ----------------------------------
  {
    // 后端接口调用频率
    useInterval(invokeApi, apiInvokeIntervalRef.current, isDetecting, [isRoiMode, roiRect]);

    useEffect(() => {
      if (!roiCanvasRef.current) return;

      let animationId: NodeJS.Timeout;

      // 截取 ROI 区域
      const roiCtx = roiCanvasRef.current.getContext('2d');
      // 这样是保证画图区域是完全撑满的，没有留白
      roiCanvasRef.current.width = roiRect.width;
      roiCanvasRef.current.height = roiRect.height;
      const realRect = transRectToRealSize(roiRect)

      function captureRoiFrame() {
        if (!roiCanvasRef.current || !monitorDomRef.current) return;

        roiCtx?.drawImage(
          monitorDomRef.current,
          realRect.x, realRect.y, realRect.width, realRect.height,  // 源图像区域
          0, 0, roiRect.width, roiRect.height);          // 目标画布区域

        // 此处还不能判断videoDomRef.current.paused, 因为即便播放状态下拖动进度条，也可以出现pause
        if (!isDetecting || !isRoiMode) return;

        animationId = setTimeout(captureRoiFrame, 200)
      }

      captureRoiFrame()

      return () => clearTimeout(animationId);
    }, [isDetecting, isRoiMode, roiRect]);

  }

  const renderMonitorDom = () => {
    if (monitorType === 'remote_camera') {
      if (remoteCameraLoading) {
        return <div className="w-full h-full flex justify-center items-center">
          <Spin
            tip={<div><VideoCameraOutlined /> 正在连接远程摄像头</div>}>
            <div className='w-[150px]'></div>
          </Spin>
        </div >
      } else {
        return <img
          src="/camera/video_feed"
          alt="Camera Stream"
          className="object-contain rounded-md w-full h-full max-h-full"
          ref={imgDomRef}
          onLoad={() => {
            if (imgDomRef.current) {
              monitorCanvasRef.current.width = imgDomRef.current.naturalWidth;
              monitorCanvasRef.current.height = imgDomRef.current.naturalHeight;
              initRoiRect();
            }
          }}
          onError={(e) => {
            console.log("camera error", e);
          }}
        />
      }
    } else {
      return <video
        src={videoSource}
        controls={monitorType === 'demo_video' || monitorType === 'change_video'}
        className="object-contain rounded-md w-full h-full max-h-full"
        ref={videoDomRef}
        onLoadedMetadata={() => {
          if (videoDomRef.current) {
            monitorCanvasRef.current.width = videoDomRef.current.videoWidth;
            monitorCanvasRef.current.height = videoDomRef.current.videoHeight;
            initRoiRect();
          }
        }}
      />
    }
  }

  return (
    <div className="h-screen flex flex-col p-3 bg-gray-100">
      {/* 标题栏 */}
      <div className="pb-2">
        <h1 className="text-xl font-bold text-gray-800 text-center">火灾监控系统</h1>

        <Controller
          isDetecting={isDetecting}
          setIsDetecting={setIsDetecting}
          onStopDetection={() => {
            if (socket && isSocketConnected) {
              socket.emit('stop_detection');
            }
          }}
          videoDomRef={videoDomRef}
          isRoiMode={isRoiMode}
          setIsRoiMode={setIsRoiMode}
          isSocketConnected={isSocketConnected}
          monitorType={monitorType}
          setMonitorType={setMonitorType}
          setRemoteCameraLoading={setRemoteCameraLoading}

          apiInvokeIntervalRef={apiInvokeIntervalRef}
          detectThresholdRef={detectThresholdRef}
          classifyThresholdRef={classifyThresholdRef}
        />
      </div>

      {/* 内容 grid-cols-[auto_1fr_1fr] */}
      <div className="flex-1 grid  sm:grid-cols-3 sm:grid-rows-2 gap-3 h-full sm:overflow-hidden">
        {/* 1. 监控视频 */}
        <Card title="监控视频">
          <div className="h-full"
          // style={{ border: '1px solid red' }}
          >
            {renderMonitorDom()}

            {/* 裁剪区域 */}
            {isRoiMode && <RoiArea roiRect={roiRect} setRoiRect={setRoiRect} />}
          </div>
        </Card>

        {/* 2. 识别结果 */}
        <Predict title="识别结果" data={predictResult} />

        {/* 3. 检测结果 */}
        <Card title="检测结果"
          extra={detectTime &&
            <div className="text-sm text-gray-600 ">
              处理时间: {detectTime} 毫秒
            </div>
          }
        >
          <canvas ref={detectionCanvasRef} className='w-full h-full max-h-[40vh] object-contain' />
        </Card>

        {/* 4. Roi区域 */}
        <Card title="Roi区域">
          <canvas ref={roiCanvasRef}
            className='w-full h-full max-h-[40vh] object-contain'
            style={{ display: isRoiMode ? 'block' : 'none' }}
          />
        </Card>

        {/* 5. Roi识别结果 */}
        <Predict title="Roi识别结果" data={roiPredictResult} hidden={!isRoiMode} />

        {/* 6. Roi检测结果 */}
        <Card title="Roi检测结果" extra={roiDetectTime &&
          <div className="text-sm text-gray-600 ">
            处理时间: {roiDetectTime} 毫秒
          </div>
        }>
          <canvas ref={roiDetectionCanvasRef}
            className='w-full h-full object-contain'
            style={{ display: isRoiMode ? 'block' : 'none' }}
          />
        </Card>
      </div>
    </div >
  );
}
