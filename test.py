"""Local tests for the LangChain learning assistant.

Run with:
    python test.py

These tests avoid real DeepSeek/Tavily network calls so they can run safely
before API keys are configured.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

import learning_assistant
from prompts import build_learning_prompt


class FakeChain:
    def __init__(self, response: str = "测试回答") -> None:
        self.response = response
        self.last_payload = None
        self.last_config = None

    def invoke(self, payload, config=None):
        self.last_payload = payload
        self.last_config = config
        return self.response


class LearningAssistantTest(unittest.TestCase):
    def test_prompt_contains_current_advanced_features(self) -> None:
        prompt = build_learning_prompt()
        messages = prompt.format_messages(
            level="初学者",
            goal="学习 Agent 开发",
            style="中文、通俗、带代码",
            question="ChatPromptTemplate 有什么用？",
            history=[HumanMessage(content="上一轮问题"), AIMessage(content="上一轮回答")],
            search_context="标题：LangChain 文档\n链接：https://example.com",
        )

        rendered = "\n".join(str(message.content) for message in messages)

        self.assertIn("学习者水平：初学者", rendered)
        self.assertIn("联网资料", rendered)
        self.assertIn("什么是 Runnable？", rendered)
        self.assertIn("MessagesPlaceholder 有什么用？", rendered)
        self.assertIn("上一轮问题", rendered)
        self.assertIn("ChatPromptTemplate 有什么用？", rendered)

    def test_required_env_rejects_missing_and_placeholder_values(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "缺少环境变量 DEEPSEEK_API_KEY"):
                learning_assistant.required_env("DEEPSEEK_API_KEY")

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "your_deepseek_api_key"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "缺少环境变量 DEEPSEEK_API_KEY"):
                learning_assistant.required_env("DEEPSEEK_API_KEY")

    def test_search_tavily_formats_results_without_real_network_call(self) -> None:
        fake_payload = {
            "results": [
                {
                    "title": "LangChain Tool Calling",
                    "content": "Tools let models request external actions.",
                    "url": "https://example.com/tool-calling",
                }
            ]
        }

        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=True):
            with patch.object(learning_assistant.TavilyClient, "search", return_value=fake_payload):
                result = learning_assistant.search_tavily("tool calling")

        self.assertIn("标题：LangChain Tool Calling", result)
        self.assertIn("摘要：Tools let models request external actions.", result)
        self.assertIn("链接：https://example.com/tool-calling", result)

    def test_search_tavily_returns_explainable_failure_text(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = learning_assistant.search_tavily("没有 key 的搜索")

        self.assertIn("[联网搜索失败]", result)
        self.assertIn("请基于已有知识回答", result)

    def test_run_turn_uses_search_context_and_updates_history(self) -> None:
        chain = FakeChain(response="这是模型回答")
        history = [
            HumanMessage(content=f"旧问题 {index}") if index % 2 == 0 else AIMessage(content=f"旧回答 {index}")
            for index in range(10)
        ]

        with patch.object(learning_assistant, "search_tavily", return_value="联网资料摘要"):
            answer = learning_assistant.run_turn(
                chain,
                question="  如何设计 Agent 工具调用？  ",
                level="",
                goal="",
                style="",
                history=history,
            )

        self.assertEqual(answer, "这是模型回答")
        self.assertEqual(chain.last_payload["question"], "如何设计 Agent 工具调用？")
        self.assertEqual(chain.last_payload["level"], "初学者")
        self.assertEqual(chain.last_payload["goal"], "学习 Agent 开发")
        self.assertEqual(chain.last_payload["style"], "中文、通俗、带代码")
        self.assertEqual(chain.last_payload["search_context"], "联网资料摘要")
        self.assertEqual(len(chain.last_payload["history"]), 8)
        self.assertEqual(chain.last_config["run_name"], "learning_assistant_answer")
        self.assertIn("learning-assistant", chain.last_config["tags"])
        self.assertEqual(history[-2].content, "如何设计 Agent 工具调用？")
        self.assertEqual(history[-1].content, "这是模型回答")

    def test_run_turn_rejects_empty_question(self) -> None:
        with self.assertRaisesRegex(ValueError, "技术问题不能为空"):
            learning_assistant.run_turn(FakeChain(), "   ", "初学者", "学习 Agent 开发", "中文", [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
