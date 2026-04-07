import { message } from "antd";
import { useEffect, useRef } from "react";


export function useLocalCamera(cameraOpen: boolean) {
    const videoDomRef = useRef<HTMLVideoElement>(null);
    const streamRef = useRef<MediaStream | null>(null);


    // 清理摄像头流
    const cleanupStream = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => {
                track.stop();
            });
            streamRef.current = null;
        }
        if (videoDomRef.current) {
            videoDomRef.current.srcObject = null;
        }
    };

    // 启动摄像头
    const startCamera = async () => {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            message.error("您的浏览器不支持本地摄像头功能");
            return;
        }

        try {
            // 先清理之前的流
            cleanupStream();

            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                }
            });

            if (videoDomRef.current && cameraOpen) {
                streamRef.current = stream;
                videoDomRef.current.srcObject = stream;

                // 等待视频元数据加载完成
                await new Promise<void>((resolve, reject) => {
                    const video = videoDomRef.current!;

                    const onLoadedMetadata = () => {
                        video.removeEventListener('loadedmetadata', onLoadedMetadata);
                        video.removeEventListener('error', onError);
                        resolve();
                    };

                    const onError = (e: Event) => {
                        video.removeEventListener('loadedmetadata', onLoadedMetadata);
                        video.removeEventListener('error', onError);
                        reject(new Error('视频加载失败 ' + String(e)));
                    };

                    video.addEventListener('loadedmetadata', onLoadedMetadata);
                    video.addEventListener('error', onError);

                    // 设置超时
                    setTimeout(() => {
                        video.removeEventListener('loadedmetadata', onLoadedMetadata);
                        video.removeEventListener('error', onError);
                        reject(new Error('视频加载超时'));
                    }, 10000);
                });

                // 开始播放
                try {
                    await videoDomRef.current.play();
                } catch (playError) {
                    console.warn('自动播放失败，需要用户交互:', playError);
                    // 自动播放失败不算错误，用户点击后会播放
                }
            }
        } catch (error) {
            console.error('摄像头启动失败:', error);

            let errorMessage = "无法访问摄像头";
            if (error instanceof Error) {
                if (error.name === 'NotAllowedError') {
                    errorMessage = "本地摄像头权限被拒绝，请在浏览器设置中允许访问摄像头";
                } else if (error.name === 'NotFoundError') {
                    errorMessage = "未找到本地摄像头设备";
                } else if (error.name === 'NotReadableError') {
                    errorMessage = "本地摄像头被其他应用占用";
                } else {
                    errorMessage = `摄像头错误: ${error.message}`;
                }
            }

            message.error(errorMessage);
        }
    };

    useEffect(() => {
        if (cameraOpen) {
            startCamera();
        } else {
            // 关闭摄像头时的清理
            cleanupStream();
        }

        // 组件卸载时的清理
        return () => {
            cleanupStream();
        };
    }, [cameraOpen]);

    return { videoDomRef };
}