import logging
import time
import numpy as np

class MockFrameReader:
    """Simulates djitellopy's FrameReader, always returning a black image."""
    def __init__(self):
        # Create a 960x720 black image (Tello standard resolution)
        self.frame = np.zeros((720, 960, 3), dtype=np.uint8)

class MockTello:
    """
    A simulated Tello class for debugging when no real drone is present.
    It implements key methods of djitellopy.Tello but only prints logs to the console.
    """
    def __init__(self):
        self.is_flying = False
        self.speed = 10
        self.battery = 100
        self.stream_on = False
        logging.info("--- MOCK TELLO INITIALIZED ---")

    def connect(self):
        logging.info("SIMULATOR: Connecting to Tello...")
        time.sleep(1)
        logging.info("SIMULATOR: Connection successful.")
        return True

    def takeoff(self):
        if not self.is_flying:
            logging.info("SIMULATOR: Takeoff command received.")
            self.is_flying = True
        else:
            logging.warning("SIMULATOR: Drone is already flying.")

    def land(self):
        if self.is_flying:
            logging.info("SIMULATOR: Land command received.")
            self.is_flying = False
        else:
            logging.warning("SIMULATOR: Drone is already on the ground.")

    def emergency(self):
        logging.critical("SIMULATOR: EMERGENCY command received! Motors stopped.")
        self.is_flying = False

    def move(self, direction, x):
        logging.info(f"SIMULATOR: Move command: {direction} for {x} cm.")

    def move_left(self, x): self.move("left", x)
    def move_right(self, x): self.move("right", x)
    def move_forward(self, x): self.move("forward", x)
    def move_back(self, x): self.move("back", x)
    def move_up(self, x): self.move("up", x)
    def move_down(self, x): self.move("down", x)

    def rotate_clockwise(self, x):
        logging.info(f"SIMULATOR: Rotate clockwise for {x} degrees.")

    def rotate_counterclockwise(self, x):
        logging.info(f"SIMULATOR: Rotate counter-clockwise for {x} degrees.")

    def set_speed(self, x):
        self.speed = x
        logging.info(f"SIMULATOR: Speed set to {x}.")

    def get_battery(self):
        logging.info(f"SIMULATOR: Querying battery. Returning {self.battery}%.")
        return self.battery

    def streamon(self):
        logging.info("SIMULATOR: Video stream ON.")
        self.stream_on = True

    def streamoff(self):
        logging.info("SIMULATOR: Video stream OFF.")
        self.stream_on = False

    def get_frame_read(self):
        logging.info("SIMULATOR: Frame reader requested.")
        return MockFrameReader()