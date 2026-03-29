from djitellopy import Tello
import logging

class DroneTools:
    """
    封装所有可供LLM调用的无人机控制工具。
    每个公开方法都是一个独立的、原子性的工具。
    """
    def __init__(self, tello_instance: Tello):
        self.tello = tello_instance

    def takeoff(self) -> str:
        """
        命令无人机从地面起飞。
        Commands the drone to take off from the ground.
        """
        try:
            # Tello SDK的takeoff是阻塞的，成功或失败后才会返回
            self.tello.takeoff()
            return "无人机已成功起飞。"
        except Exception as e:
            logging.error(f"起飞时发生错误: {e}")
            return f"起飞失败: {e}"

    def land(self) -> str:
        """
        命令无人机在地面降落。
        Commands the drone to land on the ground.
        """
        try:
            self.tello.land()
            return "无人机已成功降落。"
        except Exception as e:
            logging.error(f"降落时发生错误: {e}")
            return f"降落失败: {e}"

    def move(self, direction: str, distance_cm: int) -> str:
        """
        朝指定方向移动特定距离。
        Moves the drone in a specific direction for a certain distance.
        
        :param direction: 移动方向, 必须是 ['forward', 'back', 'left', 'right', 'up', 'down'] 中的一个。
        :param distance_cm: 移动距离（厘米）, 必须在 20 到 500 之间。
        """
        if direction not in ['forward', 'back', 'left', 'right', 'up', 'down']:
            return f"错误: 无效的方向 '{direction}'"
        if not 20 <= distance_cm <= 500:
            return f"错误: 无效的距离 {distance_cm} cm. 距离必须在 20 到 500 厘米之间。"
        
        try:
            move_func = getattr(self.tello, f"move_{direction}")
            move_func(distance_cm)
            return f"已朝 {direction} 方向移动 {distance_cm} 厘米。"
        except Exception as e:
            logging.error(f"向 {direction} 移动时发生错误: {e}")
            return f"移动失败: {e}"

    def rotate(self, direction: str, degrees: int) -> str:
        """
        顺时针或逆时针旋转无人机。
        Rotates the drone clockwise or counter-clockwise.
        
        :param direction: 旋转方向, 必须是 ['clockwise', 'counter_clockwise'] 中的一个。
        :param degrees: 旋转角度（度）, 必须在 1 到 360 之间。
        """
        if direction not in ['clockwise', 'counter_clockwise']:
            return f"错误: 无效的旋转方向 '{direction}'"
        if not 1 <= degrees <= 360:
            return f"错误: 无效的角度 {degrees}. 角度必须在 1 到 360 度之间。"
            
        try:
            if direction == 'clockwise':
                self.tello.rotate_clockwise(degrees)
            else:
                self.tello.rotate_counterclockwise(degrees)
            return f"已向 {direction} 方向旋转 {degrees} 度。"
        except Exception as e:
            logging.error(f"旋转时发生错误: {e}")
            return f"旋转失败: {e}"

    def get_battery(self) -> str:
        """
        获取无人机当前电量百分比。
        Gets the current battery percentage of the drone.
        """
        try:
            battery_level = self.tello.get_battery()
            return f"无人机当前电量为 {battery_level}%。"
        except Exception as e:
            logging.error(f"获取电量时发生错误: {e}")
            return f"获取电量失败: {e}"

    def emergency_stop(self) -> str:
        """
        立即停止无人机所有电机。仅在紧急情况下使用。
        Immediately stops all motors of the drone. Use in case of emergency.
        """
        try:
            self.tello.emergency()
            return "紧急停止已激活！所有电机已关闭。"
        except Exception as e:
            logging.error(f"执行紧急停止时发生错误: {e}")
            return f"紧急停止失败: {e}"