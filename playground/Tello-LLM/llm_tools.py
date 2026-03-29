"""
此文件包含Ollama API所需的工具定义（Tool Definitions）。
这些定义以JSON格式描述了可供大语言模型（LLM）使用的函数、
它们的参数以及功能说明。
"""

tools_definitions = [
    {
        'type': 'function',
        'function': {
            'name': 'takeoff',
            'description': '命令无人机从地面起飞。',
            'parameters': {'type': 'object', 'properties': {}}
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'land',
            'description': '命令无人机在地面降落。',
            'parameters': {'type': 'object', 'properties': {}}
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'move',
            'description': '朝指定方向移动特定距离。',
            'parameters': {
                'type': 'object',
                'properties': {
                    'direction': {
                        'type': 'string', 
                        'description': "移动方向，必须是 ['forward', 'back', 'left', 'right', 'up', 'down'] 中的一个。"
                    },
                    'distance_cm': {
                        'type': 'integer', 
                        'description': '移动距离（厘米），必须在 20 到 500 之间。'
                    }
                },
                'required': ['direction', 'distance_cm']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'rotate',
            'description': '顺时针或逆时针旋转无人机。',
            'parameters': {
                'type': 'object',
                'properties': {
                    'direction': {
                        'type': 'string',
                        'description': "旋转方向，必须是 ['clockwise', 'counter_clockwise'] 中的一个。"
                    },
                    'degrees': {
                        'type': 'integer', 
                        'description': '旋转角度（度），必须在 1 到 360 之间。'
                    }
                },
                'required': ['direction', 'degrees']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_battery',
            'description': '获取无人机当前电量百分比。',
            'parameters': {'type': 'object', 'properties': {}}
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'emergency_stop',
            'description': '紧急情况下立即停止无人机所有电机。',
            'parameters': {'type': 'object', 'properties': {}}
        }
    }
]