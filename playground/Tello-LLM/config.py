# Local model name
LLM_MODEL = "qwen3:1.7b" 

# Tello drone default waiting timeout (seconds)
TELLO_COMMAND_TIMEOUT = 15

# Whether to display real-time video. Set to False for remote debugging or environments without a graphical interface.
SHOW_VIDEO_STREAM = False

# Whether to use a real Tello drone. Set to False to use the simulator for debugging,
USE_REAL_DRONE = False

# Set to True: All instructions will be sent to the LLM for understanding.
# Set to False: The program will first try to parse the user input as a direct command (e.g., "takeoff", "move forward 50").
#              If parsing fails, it will then call the LLM. This can save time for simple commands.
ALWAYS_USE_LLM = False