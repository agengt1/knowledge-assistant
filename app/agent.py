"""
自研 Agent 循环 —— 不依赖 LangChain

面试核心亮点：
"我从零实现了 Agent 循环，替代 LangChain 的 create_agent。
好处：精确控制每次工具调用、Token 消耗、错误重试。
LangChain Agent 是黑盒，我的版本 50 行代码，逻辑完全透明。"

Agent 循环流程：
1. LLM 收到用户问题 + 工具列表
2. LLM 决定：直接回答 或 调用工具
3. 如果调用工具 → 执行 → 把结果反馈给 LLM
4. 重复，直到 LLM 给出最终回答 或 达到最大轮次
"""

import json
from openai import OpenAI


def run_agent(
    client: OpenAI,
    model: str,
    user_message: str,
    tools: list[dict],
    tool_handlers: dict,
    system_prompt: str = "你是一个智能助手，可以使用工具来回答问题。",
    max_turns: int = 3,
    history: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """
    Agent 主循环。

    参数：
        client:       OpenAI 兼容客户端
        model:        模型名
        user_message: 用户问题
        tools:        工具定义列表（OpenAI function calling 格式）
        tool_handlers: 工具名 → 执行函数 {name: callable}
        system_prompt: 系统提示
        max_turns:     最大推理轮次（防止死循环）

    返回：
        (最终回答, 执行日志)
    """
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    execution_log = []

    for turn in range(max_turns):
        # 调用 LLM
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=0.3,  # 低温度保证决策稳定
        )

        msg = response.choices[0].message

        # 如果 LLM 直接回答（不调用工具）→ 结束
        if not msg.tool_calls:
            execution_log.append({
                "turn": turn,
                "action": "answer",
                "content": msg.content,
            })
            return msg.content or "", execution_log

        # LLM 调用了工具 → 执行
        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            # 执行工具（带错误处理）
            try:
                handler = tool_handlers.get(tool_name)
                if handler is None:
                    result = f"错误：工具 '{tool_name}' 不存在"
                else:
                    result = handler(**tool_args)
            except Exception as e:
                result = f"工具执行失败：{str(e)}"

            execution_log.append({
                "turn": turn,
                "action": "tool_call",
                "tool": tool_name,
                "args": tool_args,
                "result_preview": str(result)[:200],
            })

            # 把工具调用和结果加入对话
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })

    # 达到最大轮次 → 强制结束
    execution_log.append({
        "turn": max_turns,
        "action": "force_stop",
        "content": "达到最大推理轮次",
    })
    return "抱歉，这个问题我暂时无法回答，请换个问法试试。", execution_log
