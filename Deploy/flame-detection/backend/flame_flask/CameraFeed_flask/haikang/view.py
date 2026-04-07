import sys
import os
import cv2
import numpy as np
import time
sys.path.append("/opt/MVS/Samples/64/Python")

from MvImport.MvCameraControl_class import *
from MvImport.MvErrorDefine_const import *
from MvImport.PixelType_header import *
# === 1. 把 MvImport 加进 sys.path（按你的实际路径修改） ===
# 路径是：/opt/MVS/Samples/64/Python/MvImport
mvimport_path = "/opt/MVS/Samples/64/Python/MvImport"
if mvimport_path not in sys.path:
    sys.path.append(mvimport_path)

from MvCameraControl_class import *
from MvErrorDefine_const import *

# === 2. 像 GrabImage.py 一样封装相机打开/取流 ===

def open_first_camera():
    # 枚举设备
    device_list = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
    if ret != MV_OK:
        print("EnumDevices fail! ret[0x%x]" % ret)
        return None

    if device_list.nDeviceNum == 0:
        print("No camera found!")
        return None

    print("Find %d device(s)." % device_list.nDeviceNum)

    # 这里只取第 0 台（可以做个输入选择）
    st_device_info = cast(device_list.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents

    cam = MvCamera()
    ret = cam.MV_CC_CreateHandle(st_device_info)
    if ret != MV_OK:
        print("CreateHandle fail! ret[0x%x]" % ret)
        return None

    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != MV_OK:
        print("OpenDevice fail! ret[0x%x]" % ret)
        cam.MV_CC_DestroyHandle()
        return None

    # 如果是 U3V，可以设置一下传输层缓存（可选）
    cam.MV_CC_SetEnumValue("TriggerMode", 0)  # 连续采集

    # 开始取流
    ret = cam.MV_CC_StartGrabbing()
    if ret != MV_OK:
        print("StartGrabbing fail! ret[0x%x]" % ret)
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        return None

    return cam


def frame_to_ndarray(p_data, frame_info):
    """将相机帧数据转成 numpy 数组，方便给 OpenCV 用"""
    width = frame_info.nWidth
    height = frame_info.nHeight
    pixel_type = frame_info.enPixelType
    frame_len = frame_info.nFrameLen

    # p_data 是 ctypes 的 c_ubyte 数组，这种类型可以直接给 frombuffer 用
    buf = np.frombuffer(p_data, dtype=np.uint8, count=frame_len)

    # 通常 0x01080001 是 Mono8，0x01080009 是 BayerRG8（你的是 0x1080009）
    if pixel_type == PixelType_Gvsp_Mono8:
        img = buf.reshape((height, width))
        return img

    elif pixel_type in (PixelType_Gvsp_BayerRG8,
                        PixelType_Gvsp_BayerGB8,
                        PixelType_Gvsp_BayerGR8,
                        PixelType_Gvsp_BayerBG8):
        # 先按灰度 reshape，再用 OpenCV 做 Bayer 转 BGR
        img_raw = buf.reshape((height, width))

        #  Bayer 格式大概率是 RG8，如果颜色不对再换 BG2BGR 试试
        img_bgr = cv2.cvtColor(img_raw, cv2.COLOR_BAYER_RG2BGR)
        return img_bgr

    else:
        print("Unsupported pixel type: 0x%x" % pixel_type)
        return None



def live_view():
    cam = open_first_camera()
    if cam is None:
        return

    try:
        # 申请一块 buffer 存图像
        st_frame_info = MV_FRAME_OUT_INFO_EX()
        buf_size = 40 * 1024 * 1024  # 40MB，足够一般分辨率
        p_buf = (c_ubyte * buf_size)()

        print("Press 'q' in the image window to quit.")

        while True:
            st_frame_info = MV_FRAME_OUT_INFO_EX()
            ret = cam.MV_CC_GetOneFrameTimeout(p_buf, buf_size, st_frame_info, 1000)
            if ret != MV_OK:
                # 超时或其他，不一定是错误
                # print("GetOneFrameTimeout ret[0x%x]" % ret)
                continue

            img = frame_to_ndarray(p_buf, st_frame_info)
            if img is None:
                continue

            # 如果是灰度图，OpenCV 用 imshow 也没问题
            # cv2.imshow("Hikrobot Live", img)

            # 按 q 退出
            # if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        cv2.destroyAllWindows()
        print("Camera closed.")


if __name__ == "__main__":
    live_view()
