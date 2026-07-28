import time
from djitellopy import Tello

Tello.CONTROL_UDP_PORT_CLIENT = 9000
tello = Tello("127.0.0.1")

tello.connect()
tello.takeoff()
time.sleep(3)        # 3초 대기
tello.move_forward(50)
time.sleep(3)        # 3초 대기