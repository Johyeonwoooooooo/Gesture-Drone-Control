from djitellopy import tello

drone = tello.Tello()
drone.connect()

drone.takeoff()

drone.move_forward(100)  
drone.move_up(70)
drone.move_back(100) 

drone.land()