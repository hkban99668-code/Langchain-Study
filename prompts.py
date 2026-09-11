"""Prompt definitions for the online technical learning assistant."""

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)


def build_learning_prompt() -> ChatPromptTemplate:
    system = SystemMessagePromptTemplate.from_template(
        """你是一名联网技术学习助手。
请根据学习者水平、学习目标和回答风格，用中文解释技术问题。
回答尽量使用以下结构：
1. 简单解释
2. 核心概念
3. 联网资料总结
4. 代码示例
5. 常见错误
6. 下一步练习
7. 参考来源

学习者水平：{level}
学习目标：{goal}
回答风格：{style}

联网资料可能为空。如果资料为空，明确说明本次未成功获取联网资料，
不要假装引用来源；对资料中的链接尽量保留在“参考来源”部分。"""
    )

    # Two role-correct few-shot demonstrations constrain the learning style.
    demonstrations = []
    few_shot = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template("问题：{question}"),
        ("ai", "{answer}"),
    ])
    for example in [
        {
            "question": "什么是 Runnable？",
            "answer": "简单解释：Runnable 是可以被统一调用的 LangChain 组件。\n核心概念：它支持 invoke、ainvoke 和 stream。",
        },
        {
            "question": "MessagesPlaceholder 有什么用？",
            "answer": "简单解释：它用于把运行时提供的消息列表插入 Prompt。\n核心概念：常用于注入对话历史并保留消息角色。",
        },
    ]:
        demonstrations.extend(few_shot.format_messages(**example))

    return ChatPromptTemplate(
        messages=[
            system,
            *demonstrations,
            MessagesPlaceholder("history", optional=True),
            HumanMessagePromptTemplate.from_template(
                "技术问题：{question}\n\n联网资料：\n{search_context}"
            ),
        ],
        input_variables=["level", "goal", "style", "question"],
        partial_variables={"search_context": "本次未提供联网资料。"},
        input_types={"history": list},
        validate_template=True,
    )

