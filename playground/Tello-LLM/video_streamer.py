import cv2
import threading
import time
from djitellopy import Tello
import logging

from config import SHOW_VIDEO_STREAM

class VideoStreamer:
    """在一个独立的线程中处理Tello视频流的显示。"""
    
    def __init__(self, tello_instance: Tello):
        self.tello = tello_instance
        self.frame = None
        self.running = False
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.window_name = "Tello Vision System"
        self.display_mode = "rgb"

    def _video_loop(self):
        try:
            frame_reader = self.tello.get_frame_read()
            while self.running:
                self.frame = frame_reader.frame
                if not SHOW_VIDEO_STREAM:
                    time.sleep(0.01)
                    continue
                
                if self.frame is None:
                    time.sleep(0.01)
                    continue

                display_frame = self.frame
                cv2.imshow(self.window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("检测到 'q' 键按下，将停止程序...")
                    self.running = False
                    break
        except Exception as e:
            logging.error(f"视频线程出现错误: {e}", exc_info=True)
        finally:
            if SHOW_VIDEO_STREAM:
                cv2.destroyAllWindows()

    def start(self):
        """启动视频流线程。"""
        if not self.running:
            self.running = True
            self.video_thread.start()
            logging.info("视频流处理线程已启动。")

    def stop(self):
        """停止视频流线程。"""
        if self.running:
            self.running = False
            self.video_thread.join(timeout=2)
            logging.info("视频流处理线程已停止。")