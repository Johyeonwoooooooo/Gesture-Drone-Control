import ollama
import time
import os
import argparse
import statistics

# 清除可能影响连接的代理环境变量
for proxy in ['http_proxy', 'https_proxy', 'all_proxy', 'ALL_PROXY']:
    if proxy in os.environ:
        del os.environ[proxy]

# 预设11个默认Prompt，第一个用于“热身”
PROMPTS = [
    # 1. Warm-up prompt (将被舍弃)
    "你好，请做个简单的自我介绍。",
    # 2. 知识问答
    "请详细解释一下什么是大型语言模型（LLM），并列举三个主要的应用场景。",
    # 3. 文本生成
    "写一个关于一只小猫在未来城市探险的短篇故事大纲。",
    # 4. 代码生成
    "请用 Python 写一个函数，计算一个列表中所有偶数的和。",
    # 5. 翻译任务
    "将这句话翻译成英文: '科技的进步极大地改变了人们的生活方式。'",
    # 6. 逻辑推理
    "我有三个盒子，分别标记为“苹果”、“橙子”和“苹果和橙子”。我知道所有标签都贴错了。如果我只从“苹果和橙子”盒子里拿出一个水果，看到了一个苹果，我能确定另外两个盒子里装的是什么吗？请解释你的推理过程。",
    # 7. 创意写作
    "为一款名为“星尘”的能量饮料写一句吸引人的广告语。",
    # 8. 内容摘要
    "请将以下段落总结成一句话：'人工智能（AI）是一个广泛的计算机科学领域，旨在创建能够执行通常需要人类智能的任务的机器。这些任务包括学习、推理、问题解决、感知和语言理解。AI可以分为弱AI和强AI，前者专注于执行特定任务，后者则拥有与人类相当的通用认知能力。'",
    # 9. 角色扮演
    "你现在是一位经验丰富的旅行家，请给我推荐三个适合独自背包客的亚洲国家，并说明理由。",
    # 10. 格式化输出
    "请创建一个 Markdown 表格，比较三种不同智能手机的优缺点（型号和优缺点可虚构）。",
    # 11. 诗歌创作
    "写一首关于夜晚星空的四行诗。"
]


def benchmark_ollama_model(model_name, prompt_text, verbose=False):
    """
    调用本地 Ollama 模型，统计并返回其性能指标。

    参数:
    model_name (str): 要调用的本地 Ollama 模型名称。
    prompt_text (str): 发送给模型的提示文本。
    verbose (bool): 是否打印模型的完整回复。

    返回:
    一个包含 (tokens_per_second, duration_s) 的元组，如果发生错误则返回 None。
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
            print("错误：API 响应中缺少必要的性能指标。")
            return None

        duration_ns = response.get('total_duration', 0)
        duration_s = duration_ns / 1_000_000_000

        generated_tokens = response.get('eval_count', 0)

        if duration_s > 0:
            tokens_per_second = generated_tokens / duration_s
        else:
            tokens_per_second = float('inf')

        # 如果 verbose 模式开启，则打印模型的回复
        if verbose:
            print("\n--- 模型回复 ---")
            assistant_message = response.get('message', {}).get('content', '')
            print(assistant_message)
            print("-" * 50)

        return (tokens_per_second, duration_s)

    except Exception as e:
        print(f"\n调用模型 '{model_name}' 时发生错误: {e}")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="对本地 Ollama 模型进行多轮 Prompt 基准测试。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'model_name',
        type=str,
        help="要测试的 Ollama 模型名称。\n例如: 'qwen2:7b' 或 'llama3'"
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',  # 使其成为一个开关，存在即为 True
        help="在终端打印模型的完整回答内容。"
    )
    args = parser.parse_args()

    # 用于存储每次测试结果的列表
    results_tps = []
    results_duration = []

    print(f"--- 开始对模型 '{args.model_name}' 进行基准测试 ---")
    print(f"总计将运行 {len(PROMPTS)} 个 Prompt 进行测试。")
    print("-" * 50)

    for i, prompt in enumerate(PROMPTS):
        current_prompt_num = i + 1
        print(f"\n[ {current_prompt_num}/{len(PROMPTS)} ] 正在执行 Prompt...")
        # 只打印 prompt 的前80个字符，避免刷屏
        print(f"当前 Prompt: \"{prompt[:80].replace(os.linesep, ' ')}...\"")

        result = benchmark_ollama_model(
            model_name=args.model_name,
            prompt_text=prompt,
            verbose=args.verbose
        )

        # 检查函数是否成功返回结果
        if result:
            tps, duration = result
            print(f"完成! 速率: {tps:.2f} tokens/秒, 耗时: {duration:.4f} 秒")

            if i == 0:
                print(">>> 这是模型的预热（Warm-up）运行，结果将不计入最终平均值。")
            else:
                results_tps.append(tps)
                results_duration.append(duration)
        else:
            print(f">>> Prompt {current_prompt_num} 执行失败，已跳过。")
            # 如果是第一次（热身）运行失败，直接退出以避免后续不准确
            if i == 0:
                print("!!! 预热运行失败，无法继续测试。请检查 Ollama 服务和模型名称是否正确。")
                exit(1)

    # --- 最终结果统计 ---
    if not results_tps:
        print("\n--- 测试完成，但没有收集到有效的性能数据 ---")
        print("请检查模型是否能够对后续的 Prompt 正常响应。")
    else:
        avg_tps = statistics.mean(results_tps)
        avg_duration = statistics.mean(results_duration)
        
        print("\n" + "="*50)
        print("--- 基准测试最终摘要 ---")
        print(f"测试模型: {args.model_name}")
        print(f"用于统计的有效 Prompt 数量: {len(results_tps)} (已舍弃第一次预热结果)")
        print(f"\n=> 平均 Token 生成速率: {avg_tps:.2f} tokens/秒")
        print(f"=> 平均请求响应时间: {avg_duration:.4f} 秒")
        print("="*50)