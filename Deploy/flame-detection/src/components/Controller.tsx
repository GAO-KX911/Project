import { Button, Dropdown, message, type MenuProps } from "antd";
import SettingsDrawer, { type SettingsDrawerItem } from "./SettingsDrawer";
import { useEffect, useState, type RefObject } from "react";
import { DownOutlined, SettingOutlined } from '@ant-design/icons';
import type { MenuItemType } from "antd/es/menu/interface";
import videoSource from '../assets/10.mp4';
import axios from "axios";

export default function Controller({
    isDetecting, setIsDetecting,
    isRoiMode, setIsRoiMode,
    videoDomRef,
    isSocketConnected,
    monitorType,
    setMonitorType,
    setRemoteCameraLoading,
    apiInvokeIntervalRef,
    classifyThresholdRef,
    detectThresholdRef,
}: {
    isDetecting: boolean,
    setIsDetecting: (isDetecting: boolean) => void,
    videoDomRef: RefObject<HTMLVideoElement | null>,
    isRoiMode: boolean,
    setIsRoiMode: (isRoiMode: boolean) => void,
    isSocketConnected: boolean,
    monitorType: MonitorType;
    setMonitorType: (type: MonitorType) => void;
    setRemoteCameraLoading: (loading: boolean) => void,
} & SettingsDrawerItem
) {
    const [settingsOpen, setSettingsOpen] = useState(false); // 右侧setting抽屉

    const handleDetectClick = () => {
        if (monitorType !== 'remote_camera' && videoDomRef.current && !isDetecting) {
            videoDomRef.current.play();
        }

        setIsDetecting(!isDetecting);  // 切换检测状态
    };

    // 添加处理视频更换的函数
    const handleVideoChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            const url = URL.createObjectURL(file);
            if (videoDomRef.current) {
                videoDomRef.current.src = url;
                videoDomRef.current.load();
            }
        }
    };

    const [remoteCameraAvailable, setRemoteCameraAvailable] = useState<boolean>(true);
    // 检查远程摄像头是否可用
    useEffect(() => {
        if (monitorType !== 'remote_camera') {
            return
        }
        setRemoteCameraLoading(true)
        axios<{ available: boolean, message: string }>("/camera/available").then(resp => {
            if (!resp.data.available) {
                message.error(resp.data.message)
                setRemoteCameraAvailable(false)
                setTimeout(() => {
                    message.info("将为您自动切换为本地摄像头")
                    setMonitorType("local_camera")
                    localStorage.setItem("monitorType", "local_camera");
                }, 2000)
            } else {
                // 万事俱备，自动开始检测吧
                setIsDetecting(true)
            }
        }).catch(e => {
            console.log("camera ", e);
            message.error("后端视频流服务异常，请检查是否正常运行。")
            setRemoteCameraAvailable(false)
        }).finally(() => {
            setRemoteCameraLoading(false)
        })
    }, [monitorType])

    const items: MenuProps['items'] = [
        {
            key: 'remote_camera',
            label: <a onClick={() => { }}>
                远程摄像头
            </a>,
            disabled: !remoteCameraAvailable,
        },
        {
            key: 'local_camera',
            label: '本地摄像头',
        },
        {
            key: 'demo_video',
            label: "样本视频",
            onClick: () => {
                if (videoSource && videoDomRef.current) {
                    if (monitorType === 'remote_camera') {
                        setTimeout(() => {
                            if (videoDomRef.current) {
                                videoDomRef.current.src = videoSource;
                                videoDomRef.current.load();
                            }
                        }, 100)
                    } else {
                        videoDomRef.current.src = videoSource;
                        videoDomRef.current.load();
                    }

                }
            }
        },
        {
            key: 'change_video',
            label: (
                <div>
                    <input
                        type="file"
                        accept="video/*"
                        onChange={handleVideoChange}
                        className="hidden"
                        id="video-upload"
                    />
                    <a onClick={() => document.getElementById('video-upload')?.click()}>
                        更换视频
                    </a>
                </div>
            ),
        }
    ];

    return <>
        <div className='flex justify-between flex-wrap gap-2'>
            <div className='top-6 flex gap-2 flex-wrap'>
                <Button
                    type="primary"
                    danger={isDetecting}
                    onClick={() => handleDetectClick()}
                >
                    {isDetecting ? '停止检测' : '开始检测'}
                </Button>

                <Button onClick={() => setIsRoiMode(!isRoiMode)}>
                    {isRoiMode ? '取消ROI' : '选取ROI区域'}
                </Button>

                <span>
                    <Dropdown menu={{
                        items,
                        onClick: ({ key }) => {
                            setMonitorType(key as MonitorType)
                            localStorage.setItem("monitorType", key);
                        },
                        selectedKeys: [monitorType],
                    }}>
                        <Button onClick={e => {
                            if (monitorType !== 'change_video') {
                                e.preventDefault();
                            }
                        }}>
                            {(items.find(item => item?.key === monitorType) as MenuItemType)?.label}
                            <DownOutlined />
                        </Button>
                    </Dropdown>
                </span>
            </div>

            <div className='flex items-center gap-3'>
                {!isSocketConnected && <span className='text-red-500' >WebSocket断联</span>}
                <a
                    className='cursor-pointer'
                    onClick={() => setSettingsOpen(true)}
                >
                    <SettingOutlined style={{ fontSize: 20 }} />
                </a>
            </div>

        </div>

        <SettingsDrawer
            open={settingsOpen}
            onClose={() => setSettingsOpen(false)}
            apiInvokeIntervalRef={apiInvokeIntervalRef}
            classifyThresholdRef={classifyThresholdRef}
            detectThresholdRef={detectThresholdRef}
        />
    </>
}