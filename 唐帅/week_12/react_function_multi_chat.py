"""
Function Calling ReAct agent with multi-turn conversation support.

Usage:
  python react_function_multi_chat.py
  python react_function_multi_chat.py --question "贵州茅台近一年股价表现如何？"
  python react_function_multi_chat.py --max_steps 8

Interactive commands:
  /clear   Clear conversation history
  /history Show current history size
  /exit    Exit the chat
"""

import json
import logging
import os
import time
import argparse
from dataclasses import dataclass, field
from typing import Any, Generator

from openai import OpenAI

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# client = OpenAI(
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com",
# )
# MODEL = os.getenv("AGENT_MODEL", "deepseek-v4-flash")

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MODEL = os.getenv("AGENT_MODEL", "qwen-max")

FC_SYSTEM_PROMPT = """你是一个专业的A股金融分析助手。规则：
- 调用 financial_indicator 或 stock_price 之前，必须先用 company_lookup 获取股票代码
- 数字计算必须使用 calculator 工具，不能心算
- Final Answer 必须引用具体数据来源
- 如果没有合适工具能回答，直接说明原因"""

COLORS = {
    "thought": "\033[36m",
    "action": "\033[33m",
    "obs": "\033[32m",
    "final": "\033[35m",
    "info": "\033[34m",
    "error": "\033[31m",
    "reset": "\033[0m",
}


def _c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def _assistant_message_to_dict(message: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": "assistant",
        "content": message.content or "",
    }
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in message.tool_calls
        ]
    return payload


@dataclass
class FunctionCallingChatSession:
    system_prompt: str = FC_SYSTEM_PROMPT
    messages: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.messages:
            self.reset()

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def history_size(self) -> int:
        return max(len(self.messages) - 1, 0)

    def ask(
        self, question: str, max_steps: int = 10
    ) -> Generator[dict[str, Any], None, None]:
        from tools import TOOLS_MAP, TOOLS_SCHEMA

        self.messages.append({"role": "user", "content": question})

        for step in range(1, max_steps + 1):
            response = client.chat.completions.create(
                model=MODEL,
                messages=self.messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0,
            )
            msg = response.choices[0].message
            reason = response.choices[0].finish_reason

            if reason == "stop" or not msg.tool_calls:
                answer = msg.content or "（模型返回空内容）"
                self.messages.append({"role": "assistant", "content": answer})
                yield {
                    "step": step,
                    "type": "final",
                    "thought": "",
                    "answer": answer,
                    "history_size": self.history_size(),
                }
                return

            self.messages.append(_assistant_message_to_dict(msg))

            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                tool_fn = TOOLS_MAP.get(tool_name)
                if tool_fn is None:
                    observation = f"未知工具 '{tool_name}'"
                else:
                    try:
                        observation = tool_fn(**tool_args)
                    except TypeError as exc:
                        observation = f"工具参数错误: {exc}"
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("Tool call failed: %s", tool_name)
                        observation = f"工具执行失败: {exc}"

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(observation),
                    }
                )

                yield {
                    "step": step,
                    "type": "action",
                    "thought": "",
                    "action": tool_name,
                    "action_input": tool_args,
                    "observation": str(observation),
                    "history_size": self.history_size(),
                }

        yield {
            "step": max_steps + 1,
            "type": "max_steps",
            "answer": f"已达到最大步数 {max_steps}，未能给出最终答案。",
            "history_size": self.history_size(),
        }


def run(
    question: str,
    max_steps: int = 10,
    session: FunctionCallingChatSession | None = None,
) -> Generator[dict[str, Any], None, None]:
    if session is None:
        session = FunctionCallingChatSession()
    yield from session.ask(question, max_steps=max_steps)


def run_and_print(
    question: str,
    max_steps: int = 10,
    session: FunctionCallingChatSession | None = None,
) -> FunctionCallingChatSession:
    if session is None:
        session = FunctionCallingChatSession()

    print(f"\n{'=' * 60}")
    print(f"问题: {question}")
    # print(f"模型: {MODEL}  实现: Function Calling Multi-turn")
    print(f"历史消息数: {session.history_size()}")
    print("=" * 60)

    start = time.time()

    for step_data in run(question, max_steps=max_steps, session=session):
        stype = step_data["type"]

        if stype == "action":
            print(f"\n[Step {step_data['step']}]")
            print(_c("thought", "Thought: （Function Calling 版本内部推理不可见）"))
            print(_c("action", f"Action:  {step_data['action']}"))
            print(
                _c(
                    "action",
                    f"Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}",
                )
            )
            print(_c("obs", f"Obs:     {step_data['observation'][:300]}"))

        elif stype == "final":
            elapsed = time.time() - start
            print(f"\n{'-' * 60}")
            print(_c("final", f"Final Answer:\n{step_data['answer']}"))
            print(f"\n共 {step_data['step']} 步，耗时 {elapsed:.1f}s")
            print(_c("info", f"当前历史消息数: {step_data['history_size']}"))

        elif stype in ("error", "max_steps"):
            print(_c("error", step_data.get("answer", "")))

    return session


def interactive_chat(max_steps: int = 10) -> None:
    session = FunctionCallingChatSession()

    print(f"模型: {MODEL}")
    print(
        "多轮对话已开启。输入 /clear 清空历史，输入 /history 查看历史消息数，输入 /exit 退出。"
    )

    while True:
        try:
            question = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n会话结束。")
            return

        if not question:
            continue

        command = question.lower()
        if command in {"/exit", "exit", "quit"}:
            print("会话结束。")
            return
        if command == "/clear":
            session.reset()
            print("历史已清空。")
            continue
        if command == "/history":
            print(f"当前历史消息数: {session.history_size()}")
            continue

        run_and_print(question, max_steps=max_steps, session=session)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default=None)
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()

    if args.question:
        run_and_print(args.question, args.max_steps)
    else:
        interactive_chat(args.max_steps)
