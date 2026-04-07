
import os
import cv2



# 打开视频文件
video_path_l = ['1.avi', '2.avi', '3.avi', '4.avi', '5.avi', 
        '6.avi', '7.avi', '8.avi', '9.avi', '10.avi', 
        '11.avi', '12.avi', '13.avi', '14.avi', '15.avi', 
        '16.avi', '17.avi'] 
  


for video_path in video_path_l:
    if not os.path.exists(video_path.split('.')[0]+'/'):
        os.mkdir(video_path.split('.')[0]+'/')
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # 读取视频的帧率，帧率很重要，因为它可以帮助你控制读取帧的速度
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frame per second using video: {fps}")

    # 逐帧读取视频
    frame_count = 0  # 用于计数已处理的帧数
    while True:
        ret, frame = cap.read()  # 读取一帧
        if not ret:
            break  # 如果读取失败，退出循环
        frame_count += 1
        print(os.path.join(video_path.split('.')[0], str(frame_count)))
        cv2.imwrite('%s/%s-%s.jpg'% 
                    (os.path.join(video_path.split('.')[0]),
                     os.path.join(video_path.split('.')[0]),
                     str(frame_count)), 
                     frame) 
        

    # 释放VideoCapture对象
    cap.release()
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口




