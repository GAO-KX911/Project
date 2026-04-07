import sys
import os

# 添加 MVS SDK 路径
sys.path.append("/opt/MVS/Samples/64/Python/MvImport")

from MvCameraControl_class import *
from MvErrorDefine_const import *

def detect_cameras():
    """检测所有可用的相机设备"""
    print("=== 开始检测相机设备 ===")
    
    # 枚举所有设备
    device_list = MV_CC_DEVICE_INFO_LIST()
    
    # 分别测试 USB 和 GigE 设备
    print("\n1. 检测 USB 设备...")
    ret_usb = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE, device_list)
    print(f"USB 设备枚举结果: {hex(ret_usb)}")
    
    if ret_usb == MV_OK and device_list.nDeviceNum > 0:
        print(f"找到 {device_list.nDeviceNum} 个 USB 设备")
        for i in range(device_list.nDeviceNum):
            device_info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            print(f"  USB 设备 {i}: {device_info}")
    else:
        print("未找到 USB 设备")
    
    print("\n2. 检测 GigE 设备...")
    ret_gige = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE, device_list)
    print(f"GigE 设备枚举结果: {hex(ret_gige)}")
    
    if ret_gige == MV_OK and device_list.nDeviceNum > 0:
        print(f"找到 {device_list.nDeviceNum} 个 GigE 设备")
        for i in range(device_list.nDeviceNum):
            device_info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            print(f"  GigE 设备 {i}: {device_info}")
    else:
        print("未找到 GigE 设备")
    
    print("\n3. 检测所有设备...")
    ret_all = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
    print(f"所有设备枚举结果: {hex(ret_all)}")
    
    if ret_all == MV_OK and device_list.nDeviceNum > 0:
        print(f"总共找到 {device_list.nDeviceNum} 个设备")
        for i in range(device_list.nDeviceNum):
            device_info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if device_info.nTLayerType == MV_USB_DEVICE:
                dev_type = "USB"
            elif device_info.nTLayerType == MV_GIGE_DEVICE:
                dev_type = "GigE"
            else:
                dev_type = "Unknown"
            print(f"  设备 {i}: 类型={dev_type}")
    else:
        print("未找到任何设备")
        print("可能的原因：")
        print("1. 相机未正确连接")
        print("2. 相机电源未打开")
        print("3. USB 线缆问题")
        print("4. 驱动未安装")
        print("5. 权限不足")

if __name__ == "__main__":
    detect_cameras()

