# 联网技术学习助手

本项目复用根目录 `.env` 中已有的 `ChatDeepSeek` 和 `TavilyClient` 配置。入口仍是 `main.py`。

## 运行

```bash
cd /Users/gu/langchain-agent
cp .env.example .env
# 编辑 .env，填写 DEEPSEEK_API_KEY、TAVILY_API_KEY；需要追踪时填写 LANGSMITH_API_KEY
/opt/homebrew/Caskroom/miniforge/base/envs/langchain1.2/bin/python main.py
```

## 设计

- `SystemMessagePromptTemplate`：定义助手职责、回答结构和资料使用规则。
- `HumanMessagePromptTemplate`：组织当前问题和用户画像。
- `MessagesPlaceholder`：运行时注入 `history`，保留最近四轮对话。
- `ChatPromptTemplate`：组合系统消息、few-shot、历史和当前问题。
- `partial_variables`：为联网资料提供安全默认值，搜索成功时由运行时覆盖。
- 两个 few-shot 示例：约束解释风格和输出粒度。
- `prompt | model | StrOutputParser()`：构成可追踪的 Runnable 链。

LangSmith 使用 `LANGSMITH_TRACING=true`、`LANGSMITH_API_KEY` 和 `LANGSMITH_PROJECT`。链调用设置了 `run_name` 和 tags；LangChain 的模型链和 Tavily 调用会出现在同一调用追踪中。

## 检查

```bash
cd /Users/gu/langchain-agent
PYTHONPYCACHEPREFIX=/tmp/agent_learning_assistant_pycache \
/opt/homebrew/Caskroom/miniforge/base/envs/langchain1.2/bin/python -m py_compile main.py learning_assistant.py prompts.py
```

