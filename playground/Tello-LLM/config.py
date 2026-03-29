# 本地模型名
LLM_MODEL = "qwen3:1.7b" 

# Tello无人机的默认等待超时时间（秒）
TELLO_COMMAND_TIMEOUT = 15

# 是否显示实时视频画面。在远程调试或无图形界面的环境下，请设置为 False
SHOW_VIDEO_STREAM = False

# 是否使用真实的Tello无人机。设置为 False 时，将使用模拟器进行调试，
USE_REAL_DRONE = False

# 设置为 True: 所有指令都将发送给LLM进行理解。
# 设置为 False: 程序会先尝试将用户输入作为直接指令（如 "takeoff", "move forward 50"）进行解析。
#              如果解析失败，才会调用LLM。这可以为简单指令节省时间。
ALWAYS_USE_LLM = False