"""
run_weather_tool_loop.py — Function Call 多轮工具循环（仅天气工具版）

教学重点：
  1. 模型只能看到 get_weather，一个文件即可观察完整的工具调用循环
  2. 模型可以先查询一个城市，再根据结果决定下一轮查询哪个城市
  3. 每轮工具结果都以 role=tool 回填，模型不调用工具时循环结束
  4. 最多执行 8 个工具轮次，达到上限后不再提供 tools，强制模型收尾

使用方式：
  python run_weather_tool_loop.py --question "先查宁德；若超过25°C，再查北京并比较"
  python run_weather_tool_loop.py --demo

依赖：
  pip install openai httpx
  环境变量：DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# 直接运行本文件时，确保能导入同目录的天气后端。
sys.path.insert(0, str(Path(__file__).parent))

from weather_backend import get_weather  # noqa: E402


PROVIDERS = {
    "deepseek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "dashscope": {
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
}

DEFAULT_MAX_TOOL_ROUNDS = 8


def build_client(provider: str):
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        print(f"错误：未设置 {provider.upper()}_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "查询指定城市的当前天气及未来3天预报。"
                "每次调用查询一个城市；如果任务有条件依赖，可以先查询一个城市，"
                "读取结果后再决定下一轮查询哪个城市。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市中文名，如 '宁德'、'北京'、'哈尔滨'",
                    },
                },
                "required": ["city"],
            },
        },
    },
]

TOOL_DISPATCH = {
    "get_weather": get_weather,
}

SYSTEM_PROMPT = (
    "你是一名天气助手，回答必须依据 get_weather 返回的数据。"
    "需要天气信息时调用 get_weather；如果用户提出条件式任务，先查询判断条件所需的城市，"
    "读取结果后再决定下一轮查询哪个城市。互不依赖的多个城市可以在同一轮并行调用。"
    "获得足够信息后停止调用工具并给出最终回答，不要凭空编造天气。"
)


def _execute_tool_call(tc, tool_call_log: list[dict], verbose: bool, round_number: int) -> str:
    """解析并执行一个天气工具调用；错误也作为结果回填给模型。"""
    name = tc.function.name
    raw_arguments = tc.function.arguments or "{}"

    try:
        args = json.loads(raw_arguments)
    except (json.JSONDecodeError, TypeError) as exc:
        result = f"参数 JSON 解析失败：{exc}"
        tool_call_log.append({"name": name, "args": None, "error": result})
    else:
        if not isinstance(args, dict):
            result = "参数错误：工具参数必须是 JSON 对象"
            tool_call_log.append({"name": name, "args": args, "error": result})
        else:
            tool_call_log.append({"name": name, "args": args})
            fn = TOOL_DISPATCH.get(name)
            if fn is None:
                result = f"未知工具：{name}"
            else:
                try:
                    result = fn(**args)
                except TypeError as exc:
                    result = f"参数错误：{exc}"
                except Exception as exc:
                    result = f"工具执行失败：{exc}"

    if result is None:
        result = ""
    elif not isinstance(result, str):
        result = str(result)

    if verbose:
        logged_args = tool_call_log[-1].get("args")
        print(f"  → [第 {round_number} 轮天气工具] {name}({logged_args})")
        preview = result[:120].replace("\n", " ")
        suffix = "..." if len(result) > 120 else ""
        print(f"    ↩ {preview}{suffix}\n")

    return result


def run(
    client,
    model: str,
    question: str,
    verbose: bool = True,
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
) -> dict:
    """持续执行天气工具，直到模型回答或达到工具轮次上限。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []
    tool_rounds = 0
    stop_reason = "completed"
    answer = ""

    for round_number in range(1, max_tool_rounds + 1):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            answer = msg.content or ""
            break

        tool_rounds = round_number
        messages.append(msg)

        for tc in msg.tool_calls:
            result = _execute_tool_call(tc, tool_call_log, verbose, round_number)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
    else:
        # 最后一轮工具结果已经回填；省略 tools，要求模型只生成最终文本。
        stop_reason = "max_tool_rounds"
        resp = client.chat.completions.create(model=model, messages=messages)
        msg = resp.choices[0].message
        answer = msg.content or "工具调用已达到上限，模型未生成最终回答。"

    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 最终回答（天气工具 {tool_rounds} 轮，{elapsed:.1f}s）")

    return {
        "answer": answer,
        "tool_calls": tool_call_log,
        "rounds": tool_rounds,
        "stop_reason": stop_reason,
        "elapsed": elapsed,
    }


DEMO_QUESTIONS = [
    "宁德现在天气怎么样？",
    "同时查询宁德和北京的天气并比较。",
    "先查询宁德天气；如果当前温度超过25°C，再查询北京天气并比较，否则查询哈尔滨天气并比较。",
]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Function Call 多轮工具循环（仅天气工具版）")
    parser.add_argument("--question", "-q", help="单个天气问题")
    parser.add_argument("--demo", action="store_true", help="运行内置示例")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    if not args.json:
        print(f"[Function Call Loop/Weather] provider={args.provider} model={model}\n")

    questions = DEMO_QUESTIONS if args.demo else [args.question or DEMO_QUESTIONS[0]]
    results = []
    for index, question in enumerate(questions, 1):
        if not args.json:
            print("=" * 60)
            print(f"Q{index}：{question}")
            print("=" * 60)

        result = run(client, model, question, verbose=not (args.quiet or args.json))
        result["question"] = question
        results.append(result)

        if not args.json:
            print("\n最终回答：")
            print(result["answer"])
            print()

    if args.json:
        output = results[0] if len(results) == 1 else results
        print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
