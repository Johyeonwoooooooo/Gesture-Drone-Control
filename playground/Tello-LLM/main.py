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

# --- 根据配置选择导入 Tello 或 MockTello ---
if USE_REAL_DRONE:
    from djitellopy import Tello
else:
    from mock_tello import MockTello as Tello 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 清空代理
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
                logging.warning(f"指令 '{command}' 不需要参数。")
                return None
        elif command == "move":
            if len(parts) == 3:
                direction = parts[1]
                distance = int(parts[2])
                return drone_tools.move(direction, distance)
            else:
                logging.warning("Move 指令格式错误，应为: move <direction> <distance>")
                return None
        elif command == "rotate":
            if len(parts) == 3:
                direction = parts[1]
                degrees = int(parts[2])
                return drone_tools.rotate(direction, degrees)
            else:
                logging.warning("Rotate 指令格式错误，应为: rotate <direction> <degrees>")
                return None
        else:
            return None
    except (ValueError, IndexError) as e:
        logging.warning(f"直接指令解析失败: {e}. 将交由LLM处理。")
        return None


def log_token_rate(response: dict, call_description: str):
    """从Ollama响应中计算并记录token生成速率。"""
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
            logging.info(f"LLM 性能 ({call_description}): 无法获取速率统计。")
            
    except Exception as e:
        logging.warning(f"计算 token 速率时出错: {e}")


def main():
    tello = Tello()
    video_streamer = None
    if USE_REAL_DRONE:
        logging.info("--- 运行模式: 真实无人机 ---")
    else:
        logging.info("--- 运行模式: 模拟器调试 ---")
    if not ALWAYS_USE_LLM:
        logging.info("--- 指令模式: 混合模式 (优先直接执行) ---")
    else:
        logging.info("--- 指令模式: LLM模式 (所有指令经由大模型) ---")
        
    try:
        logging.info("正在连接到 Tello 无人机...")
        tello.connect()
        tello.set_speed(30)
        if USE_REAL_DRONE:
            tello.RESPONSE_TIMEOUT = TELLO_COMMAND_TIMEOUT
        logging.info("连接成功！")
        logging.info(f"无人机电量: {tello.get_battery()}%")
        tello.streamon()
        video_streamer = VideoStreamer(tello)
        video_streamer.start()
        if SHOW_VIDEO_STREAM:
            logging.info("视频流已启动。在弹出的窗口中按 'q' 键可随时退出程序。")
        else:
            logging.info("视频流处理已在后台启动（不显示画面）。")
        drone_tools = DroneTools(tello)
        messages = [{'role': 'system', 'content': '你是一个专业的无人机控制助手。根据用户的指令，调用合适的工具来精确控制无人机。你的回答应该简洁并确认执行的动作。'}]

        # --- 主交互循环 ---
        while video_streamer.running:
            prompt = input(">>> 请输入指令 (输入 'quit' 退出): ")
            if prompt.lower() in ['quit', 'exit']:
                break
            
            direct_result = None
            if not ALWAYS_USE_LLM:
                direct_result = try_direct_command_execution(prompt, drone_tools)
            
            if direct_result:
                logging.info(f"直接指令执行结果: {direct_result}")
                continue
            
            logging.info("指令无法直接解析，转交LLM处理...")
            messages.append({'role': 'user', 'content': prompt})

            try:
                # 调用Ollama LLM
                response = ollama.chat(
                    model=LLM_MODEL,
                    messages=messages,
                    tools=tools_definitions
                )
                log_token_rate(response, "工具决策")
                
                response_message = response['message']
                messages.append(response_message)
                
                if not response_message.get('tool_calls'):
                    logging.info(f"LLM: {response_message['content']}")
                    continue

                # 执行工具调用
                for tool_call in response_message['tool_calls']:
                    func_name = tool_call['function']['name']
                    args = tool_call['function']['arguments']
                    logging.info(f"LLM 正在尝试调用工具: `{func_name}` 参数: {args}")
                    if hasattr(drone_tools, func_name):
                        tool_func = getattr(drone_tools, func_name)
                        result = tool_func(**args)
                        logging.info(f"工具执行结果: {result}")
                        messages.append({
                            'role': 'tool',
                            'content': result,
                            'tool_call_id': tool_call.get('id', '')
                        })
                    else:
                        logging.error(f"错误: LLM 尝试调用一个不存在的工具 '{func_name}'")
                        messages.append({
                            'role': 'tool',
                            'content': f"错误: 工具 '{func_name}' 不存在。",
                            'tool_call_id': tool_call.get('id', '')
                        })
                
                # 让LLM根据工具执行结果进行总结
                final_response = ollama.chat(model=LLM_MODEL, messages=messages)
                log_token_rate(final_response, "执行总结")
                logging.info(f"LLM: {final_response['message']['content']}")
                messages.append(final_response['message'])

            except Exception as e:
                logging.error(f"与LLM交互或执行工具时出错: {e}")
                logging.error(traceback.format_exc())

    except (KeyboardInterrupt, SystemExit):
        logging.info("程序被用户中断。")
    except Exception as e:
        logging.critical(f"发生致命错误: {e}")
        logging.critical(traceback.format_exc())
    finally:
        logging.info("开始执行清理程序...")
        if video_streamer and video_streamer.running:
            video_streamer.stop()

        if USE_REAL_DRONE and 'tello' in locals() and tello.is_flying:
            logging.warning("无人机仍在空中，正在执行自动降落...")
            try:
                tello.land()
            except Exception as e:
                logging.error(f"自动降落失败，尝试紧急停止: {e}")
                tello.emergency()
        
        if 'tello' in locals():
            tello.streamoff()
            
        logging.info("程序已安全退出。")


if __name__ == "__main__":
    main()