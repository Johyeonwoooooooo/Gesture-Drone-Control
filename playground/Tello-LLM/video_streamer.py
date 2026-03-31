import cv2
import threading
import time
from djitellopy import Tello
import logging

from config import SHOW_VIDEO_STREAM

class VideoStreamer:
    """Processes the display of the Tello video stream in an independent thread."""
    
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
                    print("'q' key detected, program will stop...")
                    self.running = False
                    break
        except Exception as e:
            logging.error(f"Error in video thread: {e}", exc_info=True)
        finally:
            if SHOW_VIDEO_STREAM:
                cv2.destroyAllWindows()

    def start(self):
        """Starts the video stream thread."""
        if not self.running:
            self.running = True
            self.video_thread.start()
            logging.info("Video stream processing thread started.")

    def stop(self):
        """Stops the video stream thread."""
        if self.running:
            self.running = False
            self.video_thread.join(timeout=2)
            logging.info("Video stream processing thread stopped.")