"""
This file contains the tool definitions required by the Ollama API.
These definitions describe the functions available to the Large Language Model (LLM)
in JSON format, including their parameters and functional descriptions.
"""

tools_definitions = [
    {
        'type': 'function',
        'function': {
            'name': 'takeoff',
            'description': 'Command the drone to take off from the ground.',
            'parameters': {'type': 'object', 'properties': {}}
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'land',
            'description': 'Command the drone to land on the ground.',
            'parameters': {'type': 'object', 'properties': {}}
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'move',
            'description': 'Move in a specified direction for a specific distance.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'direction': {
                        'type': 'string', 
                        'description': "Direction of movement, must be one of ['forward', 'back', 'left', 'right', 'up', 'down']."
                    },
                    'distance_cm': {
                        'type': 'integer', 
                        'description': 'Moving distance in centimeters, must be between 20 and 500.'
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
            'description': 'Rotate the drone clockwise or counter-clockwise.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'direction': {
                        'type': 'string',
                        'description': "Rotation direction, must be one of ['clockwise', 'counter_clockwise']."
                    },
                    'degrees': {
                        'type': 'integer', 
                        'description': 'Rotation angle in degrees, must be between 1 and 360.'
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
            'description': 'Get the current battery percentage of the drone.',
            'parameters': {'type': 'object', 'properties': {}}
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'emergency_stop',
            'description': 'Immediately stop all drone motors in an emergency.',
            'parameters': {'type': 'object', 'properties': {}}
        }
    }
]