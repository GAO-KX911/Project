"""
视频录制管理器
用于在检测到火焰时录制视频片段
"""
import os
import cv2
import numpy as np
import base64
import time
import subprocess
from io import BytesIO
from PIL import Image
from datetime import datetime
from threading import Lock
from collections import defaultdict


class VideoRecorder:
    """视频录制管理器"""
    
    def __init__(self, output_dir='./recordings', fps=10, max_no_flame_frames=30):
        """
        初始化视频录制器
        
        Args:
            output_dir: 视频保存目录
            fps: 视频帧率
            max_no_flame_frames: 连续多少帧没有火焰后停止录制
        """
        self.output_dir = output_dir
        self.fps = fps
        self.max_no_flame_frames = max_no_flame_frames
        
        # 为每个会话(sid)维护独立的录制状态
        self.recordings = defaultdict(lambda: {
            'video_writer': None,          # OpenCV 的视频写入器对象: None表示未开始录制，对象表示正在录制
            'frames': [],                  # 暂存的视频帧列表
            'start_time': None,            # 录制开始时间：计算录制时长，生成包含时间戳的文件名
            'no_flame_frame_count': 0,     # 连续无火焰帧计数器
            'is_recording': False,         # 是否正在录制
            'video_path': None,            # 视频文件保存路径
            'width': None,                 # 视频宽度
            'height': None,                # 视频高度
            'frame_count': 0,              # 已录制帧数
            'lock': Lock(),                # 线程锁（防止并发问题）
            'recording_enabled': True       # 是否允许开始新录制（停止检测后设为False）
        })
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    def add_frame(self, sid, pic_base64, has_flame):
        """
        添加一帧到录制缓冲区
        
        Args:
            sid: 会话ID
            detection_type: 检测类型 ('original' 或 'roi')
            pic_base64: base64编码的图像数据
            has_flame: 是否检测到火焰
        """
        recording = self.recordings[sid]

        with recording['lock']:
            try:
                # 解码图像
                image_bytes = base64.b64decode(pic_base64)
                image = Image.open(BytesIO(image_bytes))
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                height, width = cv_image.shape[:2]
                
                # 如果是第一次录制，初始化视频写入器
                # 检查是否允许开始新录制（停止检测后不允许）
                if not recording['is_recording'] and has_flame and recording['recording_enabled']:
                    # 开始新的录制
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"flame_{timestamp}_{sid[:3]}.mp4"
                    video_path = os.path.join(self.output_dir, filename)
                    
                    # 使用最简单的 mp4v 编码器（通常都可用）
                    # 录制完成后会用 ffmpeg 转换为 H.264 确保浏览器兼容
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))
                    
                    if not video_writer or not video_writer.isOpened():
                        print(f"无法创建视频写入器: {video_path}")
                        return False
                    
                    recording['video_writer'] = video_writer
                    recording['video_path'] = video_path
                    recording['start_time'] = time.time()
                    recording['is_recording'] = True
                    recording['width'] = width
                    recording['height'] = height
                    recording['frame_count'] = 0
                    recording['no_flame_frame_count'] = 0
                    print(f"开始录制视频: {video_path}")
                
                # 如果正在录制，添加帧
                if recording['is_recording']:
                    if recording['video_writer'].isOpened():
                        recording['video_writer'].write(cv_image)
                        recording['frame_count'] += 1

                    if has_flame:
                        # 检测到火焰，重置计数器并写入帧
                        recording['no_flame_frame_count'] = 0                        
                    else:
                        # 没有火焰，增加计数器
                        recording['no_flame_frame_count'] += 1
                        # 如果连续N帧没有火焰，停止录制
                        if recording['no_flame_frame_count'] >= self.max_no_flame_frames:
                            self._stop_recording(sid)
                
                return True
                
            except Exception as e:
                print(f"录制帧时出错: {e}")
                return False
    
    def _convert_to_h264(self, input_path, output_path):
        """
        使用 ffmpeg 将视频转换为 H.264 格式（浏览器兼容）
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            
        Returns:
            bool: 转换是否成功
        """
        try:
            # 检查 ffmpeg 是否可用
            subprocess.run(['ffmpeg', '-version'], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL, 
                         check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("ffmpeg 未安装或不可用，跳过视频转换")
            return False
        
        try:
            # 使用 ffmpeg 转换为 H.264（浏览器兼容格式）
            # 关键参数说明：
            # -vcodec libx264: 使用 H.264 视频编码
            # -pix_fmt yuv420p: 确保像素格式兼容（浏览器必需）
            # -movflags +faststart: 将 moov atom 移到文件开头（支持流式播放）
            # -f lavfi -i anullsrc: 添加静音音频流（确保浏览器兼容）
            # -shortest: 确保音频和视频长度一致
            # -f mp4: 强制输出 MP4 格式
            cmd = [
                'ffmpeg', 
                '-i', input_path,
                '-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',  # 添加静音音频流
                '-vcodec', 'libx264',
                '-pix_fmt', 'yuv420p',  # 浏览器兼容的像素格式（必需）
                '-acodec', 'aac',  # 音频编码为 AAC
                '-movflags', '+faststart',  # 优化浏览器播放（将 moov atom 移到开头）
                '-shortest',  # 确保音频和视频长度一致
                '-f', 'mp4',
                '-y',  # 覆盖输出文件
                output_path
            ]
            result = subprocess.run(cmd, 
                                  stdout=subprocess.DEVNULL, 
                                  stderr=subprocess.PIPE,
                                  check=True)
            
            # 检查转换是否成功
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return True
            return False
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg 转换失败: {e.stderr.decode('utf-8', errors='ignore')}")
            return False
        except Exception as e:
            print(f"视频转换出错: {e}")
            return False
    
    def _stop_recording(self, sid):
        """停止录制并保存视频"""
        recording = self.recordings[sid]
        
        if recording['is_recording'] and recording['video_writer']:
            try:
                recording['video_writer'].release()
            except Exception as e:
                print(f"释放视频写入器时出错: {e}")
            
            duration = time.time() - recording['start_time'] if recording['start_time'] else 0
            
            # 确保文件路径存在且有效
            if recording['video_path'] and os.path.exists(recording['video_path']):
                file_size = os.path.getsize(recording['video_path'])
                
                # 检查文件大小，即使帧数为0也可能有文件头，设置最小有效大小为1KB
                if file_size > 1024:  # 至少1KB才认为是有效视频
                    # 使用 ffmpeg 转换为 H.264（浏览器兼容）
                    original_path = recording['video_path']
                    temp_path = original_path + '.tmp'
                    
                    if self._convert_to_h264(original_path, temp_path):
                        # 转换成功，替换原文件
                        try:
                            if os.path.exists(temp_path):
                                os.replace(temp_path, original_path)
                                file_size = os.path.getsize(original_path)
                                print(f"视频已转换为 H.264: {original_path}")
                        except Exception as e:
                            print(f"替换视频文件失败: {e}")
                    
                    print(f"录制完成: {recording['video_path']}, "
                          f"时长: {duration:.2f}秒, "
                          f"帧数: {recording['frame_count']}, "
                          f"大小: {file_size / 1024 / 1024:.2f}MB")
                else:
                    # 如果文件太小，可能是无效视频，删除
                    try:
                        os.remove(recording['video_path'])
                        print(f"删除无效视频（文件太小）: {recording['video_path']}")
                    except Exception as e:
                        print(f"删除无效视频时出错: {e}")
            elif recording['frame_count'] == 0:
                # 如果从来没有写入过帧，清理可能的残留文件
                if recording['video_path'] and os.path.exists(recording['video_path']):
                    try:
                        os.remove(recording['video_path'])
                        print(f"删除空视频文件: {recording['video_path']}")
                    except Exception as e:
                        print(f"删除空视频文件时出错: {e}")
            
            # 重置状态（但保留video_path用于调试）
            recording['video_writer'] = None
            recording['is_recording'] = False
            recording['start_time'] = None
            recording['no_flame_frame_count'] = 0
            # 注意：不清除 frame_count，保留用于日志
    
    def stop_all_recordings(self, sid):
        """强制停止指定会话的所有录制，并禁止开始新录制"""
        if sid in self.recordings:
            # 禁止开始新录制
            self.recordings[sid]['recording_enabled'] = False
            # 停止当前录制
            self._stop_recording(sid)
    
    def stop_all_sessions(self):
        """停止所有会话的录制并保存视频"""
        for sid in list(self.recordings.keys()):
            if self.recordings[sid]['is_recording']:
                self._stop_recording(sid)
    
    def cleanup_session(self, sid):
        """清理会话的录制资源"""
        self.stop_all_recordings(sid)
        if sid in self.recordings:
            del self.recordings[sid]
    
    def enable_recording(self, sid):
        """允许指定会话开始新录制（重新开始检测时调用）"""
        if sid in self.recordings:
            self.recordings[sid]['recording_enabled'] = True
    
    def _get_video_info(self, video_path):
        """
        获取视频信息（时长和帧率）
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            dict: 包含 duration（秒）和 fps 的字典，如果解析失败返回 None
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            # 获取帧率
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 获取总帧数
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 计算时长（秒）
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                'duration': round(duration, 2),  # 保留2位小数
                'fps': round(fps, 2) if fps > 0 else None
            }
        except Exception as e:
            print(f"解析视频信息失败 {video_path}: {e}")
            return None
    
    def get_video_list(self, limit=100):
        """
        获取录制的视频列表
        
        Args:
            limit: 返回的最大视频数量
            
        Returns:
            视频信息列表
        """
        videos = []
        
        if not os.path.exists(self.output_dir):
            return videos
        
        try:
            files = os.listdir(self.output_dir)
            # 只支持 MP4 格式
            video_files = [f for f in files if f.endswith('.mp4')]
            
            # 按修改时间排序（最新的在前）
            video_files.sort(key=lambda f: os.path.getmtime(
                os.path.join(self.output_dir, f)
            ), reverse=True)
            
            for video_file in video_files[:limit]:
                video_path = os.path.join(self.output_dir, video_file)
                stat = os.stat(video_path)
                
                # 获取视频信息（时长和帧率）
                video_info = self._get_video_info(video_path)
                
                videos.append({
                    'filename': video_file,
                    'path': video_path,
                    'size': stat.st_size,
                    'created_time': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': video_info['duration'] if video_info else None,
                    'fps': video_info['fps'] if video_info else None
                })
        except Exception as e:
            print(f"获取视频列表时出错: {e}")
        
        return videos


# 全局视频录制器实例
_video_recorder = None

def get_video_recorder(output_dir='./recordings', fps=10, max_no_flame_frames=30):
    """获取全局视频录制器实例"""
    global _video_recorder
    if _video_recorder is None:
        _video_recorder = VideoRecorder(output_dir, fps, max_no_flame_frames)
    else:
        if _video_recorder.fps != fps:
            print(f"fps 发生变化 ({_video_recorder.fps} -> {fps})，保存当前视频并重新初始化录制器")
            # 停止所有正在录制的会话并保存视频
            _video_recorder.stop_all_sessions()
            # 创建新的录制器实例
            _video_recorder = VideoRecorder(output_dir, fps, max_no_flame_frames)
    return _video_recorder


