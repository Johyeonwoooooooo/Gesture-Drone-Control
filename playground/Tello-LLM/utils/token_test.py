import ollama
import time
import os
import argparse
import statistics

# Clear proxy environment variables that may affect connection
for proxy in ['http_proxy', 'https_proxy', 'all_proxy', 'ALL_PROXY']:
    if proxy in os.environ:
        del os.environ[proxy]

# Preset 11 default prompts, the first one for "warm-up"
PROMPTS = [
    # 1. Warm-up prompt (will be discarded)
    "Hello, please give a brief introduction of yourself.",
    # 2. Knowledge Q&A
    "Please explain in detail what a Large Language Model (LLM) is and list three main application scenarios.",
    # 3. Text Generation
    "Write a short story outline about a kitten's adventure in a futuristic city.",
    # 4. Code Generation
    "Please write a function in Python to calculate the sum of all even numbers in a list.",
    # 5. Translation Task
    "Translate this sentence into English: 'The advancement of technology has greatly changed people's way of life.'",
    # 6. Logical Reasoning
    "I have three boxes labeled \"Apples\", \"Oranges\", and \"Apples and Oranges\". I know all the labels are wrong. If I take one fruit from the \"Apples and Oranges\" box and see an apple, can I determine what is in the other two boxes? Please explain your reasoning process.",
    # 7. Creative Writing
    "Write an eye-catching slogan for an energy drink named \"Stardust\".",
    # 8. Summarization
    "Please summarize the following paragraph into one sentence: 'Artificial Intelligence (AI) is a broad field of computer science aimed at creating machines capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding. AI can be divided into weak AI and strong AI, with the former focusing on specific tasks and the latter possessing general cognitive abilities comparable to humans.'",
    # 9. Role-playing
    "You are now an experienced traveler. Please recommend three Asian countries suitable for solo backpackers and explain why.",
    # 10. Formatted Output
    "Please create a Markdown table comparing the pros and cons of three different smartphones (models and pros/cons can be fictional).",
    # 11. Poetry Creation
    "Write a four-line poem about the starry sky at night."
]


def benchmark_ollama_model(model_name, prompt_text, verbose=False):
    """
    Calls the local Ollama model, statistics and returns its performance metrics.

    Parameters:
    model_name (str): The name of the local Ollama model to call.
    prompt_text (str): The prompt text sent to the model.
    verbose (bool): Whether to print the model's full response.

    Returns:
    A tuple containing (tokens_per_second, duration_s), or None if an error occurs.
    """
    try:
        start_time = time.time()
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt_text}],
            stream=False
        )
        end_time = time.time()

        if 'total_duration' not in response or 'eval_count' not in response:
            print("Error: Required performance metrics are missing from the API response.")
            return None

        duration_ns = response.get('total_duration', 0)
        duration_s = duration_ns / 1_000_000_000

        generated_tokens = response.get('eval_count', 0)

        if duration_s > 0:
            tokens_per_second = generated_tokens / duration_s
        else:
            tokens_per_second = float('inf')

        # If verbose mode is enabled, print the model's response
        if verbose:
            print("\n--- Model Response ---")
            assistant_message = response.get('message', {}).get('content', '')
            print(assistant_message)
            print("-" * 50)

        return (tokens_per_second, duration_s)

    except Exception as e:
        print(f"\nError occurred while calling model '{model_name}': {e}")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Perform multiple rounds of prompt benchmark testing on local Ollama models.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'model_name',
        type=str,
        help="The name of the Ollama model to test.\nExample: 'qwen2:7b' or 'llama3'"
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',  # Makes it a switch, True if present
        help="Print the model's full response in the terminal."
    )
    args = parser.parse_args()

    # List to store result of each test
    results_tps = []
    results_duration = []

    print(f"--- Starting benchmark testing for model '{args.model_name}' ---")
    print(f"A total of {len(PROMPTS)} prompts will be run for testing.")
    print("-" * 50)

    for i, prompt in enumerate(PROMPTS):
        current_prompt_num = i + 1
        print(f"\n[ {current_prompt_num}/{len(PROMPTS)} ] Executing Prompt...")
        # Only print the first 80 characters of the prompt to avoid screen spamming
        print(f"Current Prompt: \"{prompt[:80].replace(os.linesep, ' ')}...\"")

        result = benchmark_ollama_model(
            model_name=args.model_name,
            prompt_text=prompt,
            verbose=args.verbose
        )

        # Check if the function successfully returns a result
        if result:
            tps, duration = result
            print(f"Done! Rate: {tps:.2f} tokens/sec, Duration: {duration:.4f} sec")

            if i == 0:
                print(">>> This is a model warm-up run, results will not be included in the final average.")
            else:
                results_tps.append(tps)
                results_duration.append(duration)
        else:
            print(f">>> Prompt {current_prompt_num} failed to execute, skipped.")
            # If the first (warm-up) run fails, exit directly to avoid subsequent inaccuracies
            if i == 0:
                print("!!! Warm-up run failed, unable to continue testing. Please check if the Ollama service and model name are correct.")
                exit(1)

    # --- Final results statistics ---
    if not results_tps:
        print("\n--- Testing complete, but no valid performance data was collected ---")
        print("Please check if the model can respond correctly to subsequent prompts.")
    else:
        avg_tps = statistics.mean(results_tps)
        avg_duration = statistics.mean(results_duration)
        
        print("\n" + "="*50)
        print("--- Benchmark Final Summary ---")
        print(f"Tested Model: {args.model_name}")
        print(f"Number of valid prompts used for statistics: {len(results_tps)} (First warm-up result discarded)")
        print(f"\n=> Average Token Generation Rate: {avg_tps:.2f} tokens/sec")
        print(f"=> Average Request Response Time: {avg_duration:.4f} sec")
        print("="*50)