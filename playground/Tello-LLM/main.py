import ollama
import json
import logging
import traceback
import os
import time
from typing import Union 

from config import LLM_MODEL, TELLO_COMMAND_TIMEOUT, SHOW_VIDEO_STREAM, USE_REAL_DRONE, ALWAYS_USE_LLM
from tello_tools import DroneTools
from video_streamer import VideoStreamer
from llm_tools import tools_definitions

# --- Choose between Tello, Simulator or MockTello based on configuration ---
if USE_REAL_DRONE:
    from djitellopy import Tello
elif USE_SIMULATOR:
    from djitellopy import Tello

    Tello.CONTROL_UDP_PORT_CLIENT = 9000
    tello = Tello("127.0.0.1")  

else:
    from mock_tello import MockTello as Tello 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Clear proxies
for proxy in ['http_proxy', 'https_proxy', 'all_proxy', 'ALL_PROXY']:
    if proxy in os.environ:
        os.environ[proxy] = ''
        

def try_direct_command_execution(prompt: str, drone_tools: DroneTools) -> Union[str, None]:
    prompt = prompt.strip().lower()
    parts = prompt.split()
    if not parts:
        return None
    command = parts[0]
    try:
        if command in ["takeoff", "land", "get_battery", "emergency_stop"]:
            if len(parts) == 1:
                tool_func = getattr(drone_tools, command)
                return tool_func()
            else:
                logging.warning(f"Command '{command}' does not require parameters.")
                return None
        elif command == "move":
            if len(parts) == 3:
                direction = parts[1]
                distance = int(parts[2])
                return drone_tools.move(direction, distance)
            else:
                logging.warning("Move command format error, should be: move <direction> <distance>")
                return None
        elif command == "rotate":
            if len(parts) == 3:
                direction = parts[1]
                degrees = int(parts[2])
                return drone_tools.rotate(direction, degrees)
            else:
                logging.warning("Rotate command format error, should be: rotate <direction> <degrees>")
                return None
        else:
            return None
    except (ValueError, IndexError) as e:
        logging.warning(f"Direct command parsing failed: {e}. Passing to LLM.")
        return None


def log_token_rate(response: dict, call_description: str):
    """Calculate and log token generation rate from Ollama response."""
    try:
        if 'eval_count' in response and 'eval_duration' in response and response['eval_duration'] > 0:
            eval_count = response['eval_count']
            duration_s = response['eval_duration'] / 1_000_000_000  
            tokens_per_second = eval_count / duration_s
            # ANSI escape codes for colors
            GREEN = '\033[92m'
            ENDC = '\033[0m'
            
            logging.info(f"{GREEN}LLM 性能 ({call_description}): "
                         f"{eval_count} tokens in {duration_s:.2f}s "
                         f"-> {tokens_per_second:.2f} tokens/s{ENDC}")
        else:
            logging.info(f"LLM Performance ({call_description}): Cannot obtain rate statistics.")
            
    except Exception as e:
        logging.warning(f"Error calculating token rate: {e}")


def main():
    tello = Tello()
    video_streamer = None
    if USE_REAL_DRONE:
        logging.info("--- Operation Mode: Real Drone ---")
    else:
        logging.info("--- Operation Mode: Simulator Debugging ---")
    if not ALWAYS_USE_LLM:
        logging.info("--- Command Mode: Hybrid Mode (Prioritize direct execution) ---")
    else:
        logging.info("--- Command Mode: LLM Mode (All commands via large model) ---")
        
    try:
        logging.info("Connecting to Tello drone...")
        tello.connect()
        tello.set_speed(30)
        if USE_REAL_DRONE:
            tello.RESPONSE_TIMEOUT = TELLO_COMMAND_TIMEOUT
        logging.info("Connection successful!")
        logging.info(f"Drone battery: {tello.get_battery()}%")
        tello.streamon()
        video_streamer = VideoStreamer(tello)
        video_streamer.start()
        if SHOW_VIDEO_STREAM:
            logging.info("Video stream started. Press 'q' in the popup window to exit the program anytime.")
        else:
            logging.info("Video stream processing started in the background (no display).")
        drone_tools = DroneTools(tello)
        messages = [{'role': 'system', 'content': 'You are a professional drone control assistant. Call appropriate tools based on user instructions to precisely control the drone. Your response should be concise and confirm the executed action.'}]

        # --- Main Interaction Loop ---
        while video_streamer.running:
            prompt = input(">>> Please enter a command (type 'quit' to exit): ")
            if prompt.lower() in ['quit', 'exit']:
                break
            
            direct_result = None
            if not ALWAYS_USE_LLM:
                direct_result = try_direct_command_execution(prompt, drone_tools)
            
            if direct_result:
                logging.info(f"Direct command execution result: {direct_result}")
                continue
            
            logging.info("Command cannot be parsed directly, passing to LLM...")
            messages.append({'role': 'user', 'content': prompt})

            try:
                # 调用Ollama LLM
                response = ollama.chat(
                    model=LLM_MODEL,
                    messages=messages,
                    tools=tools_definitions
                )
                log_token_rate(response, "Tool Decision")
                
                response_message = response['message']
                messages.append(response_message)
                
                if not response_message.get('tool_calls'):
                    logging.info(f"LLM: {response_message['content']}")
                    continue

                # 执行工具调用
                for tool_call in response_message['tool_calls']:
                    func_name = tool_call['function']['name']
                    args = tool_call['function']['arguments']
                    logging.info(f"LLM is attempting to call tool: `{func_name}` Args: {args}")
                    if hasattr(drone_tools, func_name):
                        tool_func = getattr(drone_tools, func_name)
                        result = tool_func(**args)
                        logging.info(f"Tool execution result: {result}")
                        messages.append({
                            'role': 'tool',
                            'content': result,
                            'tool_call_id': tool_call.get('id', '')
                        })
                    else:
                        logging.error(f"Error: LLM tried to call a non-existent tool '{func_name}'")
                        messages.append({
                            'role': 'tool',
                            'content': f"Error: Tool '{func_name}' does not exist.",
                            'tool_call_id': tool_call.get('id', '')
                        })
                
                # Let LLM summarize based on the tool execution result
                final_response = ollama.chat(model=LLM_MODEL, messages=messages)
                log_token_rate(final_response, "Execution Summary")
                logging.info(f"LLM: {final_response['message']['content']}")
                messages.append(final_response['message'])

            except Exception as e:
                logging.error(f"Error interacting with LLM or executing tool: {e}")
                logging.error(traceback.format_exc())

    except (KeyboardInterrupt, SystemExit):
        logging.info("Program interrupted by user.")
    except Exception as e:
        logging.critical(f"Fatal error occurred: {e}")
        logging.critical(traceback.format_exc())
    finally:
        logging.info("Executing cleanup procedure...")
        if video_streamer and video_streamer.running:
            video_streamer.stop()

        if USE_REAL_DRONE and 'tello' in locals() and tello.is_flying:
            logging.warning("Drone is still flying, performing automatic landing...")
            try:
                tello.land()
            except Exception as e:
                logging.error(f"Automatic landing failed, attempting emergency stop: {e}")
                tello.emergency()
        
        if 'tello' in locals():
            tello.streamoff()
            
        logging.info("Program exited safely.")


if __name__ == "__main__":
    main()