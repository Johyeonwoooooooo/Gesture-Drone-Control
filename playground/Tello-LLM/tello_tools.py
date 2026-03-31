from djitellopy import Tello
import logging

class DroneTools:
    """
    Encapsulates all drone control tools available for LLM calls.
    Each public method is an independent, atomic tool.
    """
    def __init__(self, tello_instance: Tello):
        self.tello = tello_instance

    def takeoff(self) -> str:
        """
        Command the drone to take off from the ground.
        """
        try:
            # Tello SDK takeoff is blocking, returns after success or failure
            self.tello.takeoff()
            return "Drone has successfully taken off."
        except Exception as e:
            logging.error(f"Error during takeoff: {e}")
            return f"Takeoff failed: {e}"

    def land(self) -> str:
        """
        Command the drone to land on the ground.
        """
        try:
            self.tello.land()
            return "Drone has successfully landed."
        except Exception as e:
            logging.error(f"Error during landing: {e}")
            return f"Landing failed: {e}"

    def move(self, direction: str, distance_cm: int) -> str:
        """
        Move the drone in a specific direction for a certain distance.
        
        :param direction: Direction of movement, must be one of ['forward', 'back', 'left', 'right', 'up', 'down'].
        :param distance_cm: Movement distance (cm), must be between 20 and 500.
        """
        if direction not in ['forward', 'back', 'left', 'right', 'up', 'down']:
            return f"Error: Invalid direction '{direction}'"
        if not 20 <= distance_cm <= 500:
            return f"Error: Invalid distance {distance_cm} cm. Distance must be between 20 and 500 cm."
        
        try:
            move_func = getattr(self.tello, f"move_{direction}")
            move_func(distance_cm)
            return f"Moved {distance_cm} cm in the {direction} direction."
        except Exception as e:
            logging.error(f"Error while moving {direction}: {e}")
            return f"Movement failed: {e}"

    def rotate(self, direction: str, degrees: int) -> str:
        """
        Rotate the drone clockwise or counter-clockwise.
        
        :param direction: Rotation direction, must be one of ['clockwise', 'counter_clockwise'].
        :param degrees: Rotation angle (degrees), must be between 1 and 360.
        """
        if direction not in ['clockwise', 'counter_clockwise']:
            return f"Error: Invalid rotation direction '{direction}'"
        if not 1 <= degrees <= 360:
            return f"Error: Invalid angle {degrees}. Angle must be between 1 and 360 degrees."
            
        try:
            if direction == 'clockwise':
                self.tello.rotate_clockwise(degrees)
            else:
                self.tello.rotate_counterclockwise(degrees)
            return f"Rotated {degrees} degrees in the {direction} direction."
        except Exception as e:
            logging.error(f"Error during rotation: {e}")
            return f"Rotation failed: {e}"

    def get_battery(self) -> str:
        """
        Get the current battery percentage of the drone.
        """
        try:
            battery_level = self.tello.get_battery()
            return f"Current drone battery is {battery_level}%."
        except Exception as e:
            logging.error(f"Error while getting battery: {e}")
            return f"Failed to get battery: {e}"

    def emergency_stop(self) -> str:
        """
        Immediately stop all motors of the drone. Use in case of emergency.
        """
        try:
            self.tello.emergency()
            return "Emergency stop activated! All motors shut down."
        except Exception as e:
            logging.error(f"Error executing emergency stop: {e}")
            return f"Emergency stop failed: {e}"