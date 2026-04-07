from PySide6 import QtWidgets, QtCore, QtGui
import cv2, os, time
from threading import Thread
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 不然每次YOLO处理都会输出调试信息
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO

class MWindow(QtWidgets.QMainWindow):

    def __init__(self):

        super().__init__()

        # 设置界面
        self.setupUI()

        self.camBtn.clicked.connect(self.startCamera)
        self.stopBtn.clicked.connect(self.stop)

        # 定义定时器，用于控制显示视频的帧率
        self.timer_camera = QtCore.QTimer()
        # 定时到了，回调 self.show_camera
        self.timer_camera.timeout.connect(self.show_camera)

        # 加载 YOLO nano 模型，第一次比较耗时，要20秒左右
        self.model = YOLO('yolov8n.pt')

        # 要处理的视频帧图片队列，目前就放1帧图片
        self.frameToAnalyze = []

        # 启动处理视频帧独立线程
        Thread(target=self.frameAnalyzeThreadFunc,daemon=True).start()

    def setupUI(self):

        self.resize(1200, 800)

        self.setWindowTitle('白月黑羽 YOLO-Qt 演示')

        # central Widget
        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)

        # central Widget 里面的 主 layout
        mainLayout = QtWidgets.QVBoxLayout(centralWidget)

        # 界面的上半部分 : 图形展示部分
        topLayout = QtWidgets.QHBoxLayout()
        self.label_ori_video = QtWidgets.QLabel(self)
        self.label_treated = QtWidgets.QLabel(self)
        self.label_ori_video.setMinimumSize(520,400)
        self.label_treated.setMinimumSize(520,400)
        self.label_ori_video.setStyleSheet('border:1px solid #D7E2F9;')
        self.label_treated.setStyleSheet('border:1px solid #D7E2F9;')

        topLayout.addWidget(self.label_ori_video)
        topLayout.addWidget(self.label_treated)

        mainLayout.addLayout(topLayout)

        # 界面下半部分： 输出框 和 按钮
        groupBox = QtWidgets.QGroupBox(self)

        bottomLayout =  QtWidgets.QHBoxLayout(groupBox)
        self.textLog = QtWidgets.QTextBrowser()
        bottomLayout.addWidget(self.textLog)

        mainLayout.addWidget(groupBox)

        btnLayout = QtWidgets.QVBoxLayout()
        self.videoBtn = QtWidgets.QPushButton('视频文件')
        self.camBtn   = QtWidgets.QPushButton('摄像头')
        self.stopBtn  = QtWidgets.QPushButton('停止')
        btnLayout.addWidget(self.videoBtn)
        btnLayout.addWidget(self.camBtn)
        btnLayout.addWidget(self.stopBtn)
        bottomLayout.addLayout(btnLayout)


    def startCamera(self):

        # 参考 https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html

        # 在 windows上指定使用 cv2.CAP_DSHOW 会让打开摄像头快很多，
        # 在 Linux/Mac上 指定 V4L, FFMPEG 或者 GSTREAMER
        # self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("1号摄像头不能打开")
            return()

        if self.timer_camera.isActive() == False:  # 若定时器未启动
            self.timer_camera.start(50)


    def show_camera(self):

        ret, frame = self.cap.read()  # 从视频流中读取
        if not ret:
            return

        # 把读到的16:10帧的大小重新设置
        frame = cv2.resize(frame, (520, 400))
        # 视频色彩转换回RGB，OpenCV images as BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qImage = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0],
                                 QtGui.QImage.Format_RGB888)  # 变成QImage形式
        # 往显示视频的Label里 显示QImage
        self.label_ori_video.setPixmap(QtGui.QPixmap.fromImage(qImage))

        # 如果当前没有处理任务
        if not self.frameToAnalyze:
            self.frameToAnalyze.append(frame)

    def frameAnalyzeThreadFunc(self):

        while True:
            if not self.frameToAnalyze:
                time.sleep(0.05)
                continue

            frame = self.frameToAnalyze.pop(0)

            results = self.model(frame)[0]

            img = results.plot(line_width=1)

            qImage = QtGui.QImage(img.data, img.shape[1], img.shape[0],
                                    QtGui.QImage.Format_RGB888)  # 变成QImage形式

            self.label_treated.setPixmap(QtGui.QPixmap.fromImage(qImage))  # 往显示Label里 显示QImage

            time.sleep(0.5)

    def stop(self):
        self.timer_camera.stop()  # 关闭定时器
        self.cap.release()  # 释放视频流
        self.label_ori_video.clear()  # 清空视频显示区域
        self.label_treated.clear()  # 清空视频显示区域


app = QtWidgets.QApplication()
window = MWindow()
window.show()
app.exec()