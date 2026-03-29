# Tello LLM

This project uses a locally running large language model (via Ollama) to interpret natural language commands and control a DJI Tello drone through tools. The project also provides a unified tools interface for future functionality expansion.

This project has been tested on the following hardware environments, and testing and adaptation for more platforms will be released continuously:

|Device|Plantform|OS|LLM|
|--|--|--|--|
|Nvidia Orin DK|Arm64|Ubuntu 20.04|Qwen3:1.7b|
|Macbook Air M4|Arm64|MacOS 15.5|Qwen3:1.7b|

Some of the resources involved in this project can be obtained from the following network drive links:

[Note]: The current code does not currently use the resources linked below; they will be used later when adding depth map conversion functionality.

```bash
https://pan.baidu.com/s/1tlPzl8ecldkygwWHHuwHhw?pwd=g453
```

---
# Features

- **Natural Language Control**: Control the drone using everyday language (e.g., "fly forward 50 cm");
- **Live Video Stream**: View the drone's first-person view in real time in a separate window.
- **Local LLM Driver**: All language understanding is performed locally, eliminating the need for an internet connection, ensuring privacy and low latency.
- **Tool-Based Design**: Control commands are designed as a series of clear "tools" for easy LLM understanding and access.
- **Safety Assurance**: Includes emergency stop commands and automatically checks and lands the drone upon program exit.
- **Extensible Functionality**: Unified tool function definition interface facilitates future functionality expansion.

----
# Step 1. Hardware and Network Configuration

Before beginning, you need to confirm your hardware and network configuration:

* A working Tello drone;
* A computer with a Wi-Fi module;
* The computer must be able to connect to the wireless network provided by the Tello drone;
* The computer's `192.168.10.X` network segment does not conflict.

----
# Step 2. Install Dependencies

## 2.1 Install Ollama

If Ollama is already installed on your computer, you can skip this step.

* Ollama download link: [https://ollama.com/download](https://ollama.com/download)

Here, using the Linux platform as an example, use the following command:
```bash
$ curl -fsSL https://ollama.com/install.sh | sh
```

## 2.2 Creating a virtual environment

```bash
$ conda create -n tello python=3.8
$ conda activate tello
$ pip install -r requirements.txt
```

## 2.3 Installing dependency libraries

```bash
$ sudo apt-get install libopenblas-base
```

----
# Step 3. Pulling the model

## 3.1 Searching for a model
After successfully installing Ollama, select a suitable language model from the official website as the caller for the tools:

* Ollama model search page: [https://ollama.com/](https://ollama.com/)

Here, Taking `Qwen3:1.7b` as an example, although this model has a small number of parameters, thanks to the support of tools, it can strike a good balance between control performance and resource consumption. Of course, you can also choose a model with more parameters for performance reasons:

![qwen](./qwen.png)

Use the following command to pull the model

```bash
$ ollama pull qwen3:1.7b
```

## 3.2 Testing the Token Output Rate

Due to different hardware and software configurations, the token output rate of the same model may vary significantly. We strongly recommend using the provided script to test the token output rate on your device before officially starting:

* Prints only the test results, not the model output.
```bash
$ condo activate tello
$ python utils/token_test.py qwen3:1.7b
```

* Prints the test results and model output:
```bash
$ condo activate tello
$ python utils/token_test.py qwen3:1.7b -v
```

----
# Step 4. Configuration and Startup

## 4.1 Configuration
To make this project more enjoyable, you should configure the `config.py` file according to your needs:

```python
# Local model name
LLM_MODEL = "qwen3:1.7b"

# Default wait timeout for the Tello drone (in seconds)
TELLO_COMMAND_TIMEOUT = 15

# Whether to display the live video feed. Set this to False for remote debugging or a non-GUI environment.
SHOW_VIDEO_STREAM = False

# Whether to use a real Tello drone. If set to False, the simulator will be used for debugging.
USE_REAL_DRONE = False

# Set to True: All commands will be sent to the LLM for interpretation.
# Set to False: The program will first attempt to interpret user input as direct commands (e.g., "takeoff", "move forward 50").
# If this fails, the LLM will be called. This can save time for simple commands.
ALWAYS_USE_LLM = False
```

## 4.2 Starting the Script

Start the script in the following order:

1. **Power on the Tello drone** and wait for the indicator light to flash yellow.

2. **Connect to the drone's Wi-Fi**: On your computer, search for Wi-Fi networks and connect to the network named `TELLO-XXXXX`.

3. **Run the main program**: In your project folder, open a terminal and run:

```bash
$ condo activate tello
$ python main.py
```

### Directly Commanding the Model
After the script starts successfully, you can control the drone according to the prompts. Simply enter the command you want to tell the model in the terminal. After the model inference completes, the corresponding tool will be automatically invoked.

### Hybrid Command Mode
Thanks to engineering optimizations, you can directly enter the tools defined in `tools_definitions` in the `llm_tools.py` file in the terminal. For example, the `takeoff` command will directly take off the drone without passing it to the language model, thus reducing the drone's response time.

----
# Expanding tool functionality

[Under testing]