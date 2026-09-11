"""联网技术学习助手，复用项目根目录的 .env 配置。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from tavily import TavilyClient

from prompts import build_learning_prompt


load_dotenv(Path(__file__).with_name(".env"), override=False)


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or value.startswith(("your_", "replace_with_", "你的_")):
        raise RuntimeError(f"缺少环境变量 {name}，请复制 .env.example 为 .env 后填写。")
    return value


def search_tavily(query: str) -> str:
    """搜索失败时返回可解释文本，让模型继续基于已有知识回答。"""
    try:
        client = TavilyClient(api_key=required_env("TAVILY_API_KEY"))
        payload = client.search(query=query, max_results=5)
        results = payload.get("results", []) if isinstance(payload, dict) else []
        if not results:
            return "[联网搜索失败] Tavily 返回结果为空。请基于已有知识回答。"
        return "\n\n".join(
            f"{index}. 标题：{item.get('title', '未命名资料')}\n"
            f"摘要：{item.get('content', '')}\n"
            f"链接：{item.get('url', '')}"
            for index, item in enumerate(results, 1)
            if isinstance(item, dict)
        )
    except Exception as exc:
        return f"[联网搜索失败] {type(exc).__name__}: {exc}\n请基于已有知识回答。"


def create_chain():
    model_kwargs: dict[str, Any] = {
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        "api_key": required_env("DEEPSEEK_API_KEY"),
        "temperature": float(os.getenv("DEEPSEEK_TEMPERATURE", "0")),
        "streaming": True,
    }
    base_url = os.getenv("DEEPSEEK_BASE_URL", "").strip()
    if base_url:
        model_kwargs["base_url"] = base_url
    return build_learning_prompt() | ChatDeepSeek(**model_kwargs) | StrOutputParser()


def run_turn(chain, question: str, level: str, goal: str, style: str,
             history: list[BaseMessage]) -> str:
    question = question.strip()
    if not question:
        raise ValueError("技术问题不能为空。")
    answer = chain.invoke(
        {
            "question": question,
            "level": level.strip() or "初学者",
            "goal": goal.strip() or "学习 Agent 开发",
            "style": style.strip() or "中文、通俗、带代码",
            "history": history[-8:],
            "search_context": search_tavily(question),
        },
        config={"run_name": "learning_assistant_answer", "tags": ["learning-assistant"]},
    )
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("模型返回结果为空或格式异常。")
    history.extend([HumanMessage(content=question), AIMessage(content=answer)])
    return answer


def main() -> None:
    chain = create_chain()
    history: list[BaseMessage] = []
    print("联网技术学习助手已启动，输入 quit 退出。")
    level = input("用户水平（默认初学者）：").strip() or "初学者"
    goal = input("学习目标（默认学习 Agent 开发）：").strip() or "学习 Agent 开发"
    style = input("回答风格（默认中文、通俗、带代码）：").strip() or "中文、通俗、带代码"
    while True:
        question = input("\n技术问题：").strip()
        if question.lower() in {"quit", "exit", "退出"}:
            break
        try:
            print("\n" + run_turn(chain, question, level, goal, style, history))
        except Exception as exc:
            print(f"\n运行失败：{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()

