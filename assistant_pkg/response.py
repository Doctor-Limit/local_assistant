# response.py
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class ResponseGenerator:
    def __init__(self, config, tool_functions=None, llm_engine=None, context_holder=None):
        self.config = config
        self.tool_functions = {func.__name__: func for func in (tool_functions or [])}
        self.llm_engine = llm_engine
        self.context_holder = context_holder  # 用于记录工具调用
        self.max_tool_rounds = 3  # 最大工具调用轮数，防止无限循环

    def generate(self, llm_response: dict, messages: List[dict]) -> Tuple[str, List[dict]]:
        """
        处理 LLM 响应，支持多轮工具调用。
        返回最终文本和更新后的消息列表。
        """
        current_message = llm_response.get("message", {})
        tool_round = 0

        # 循环处理工具调用，直到没有工具调用或达到最大轮数
        while "tool_calls" in current_message and current_message["tool_calls"] and tool_round < self.max_tool_rounds:
            tool_round += 1
            logger.debug(f"第 {tool_round} 轮工具调用")

            for tool_call in current_message["tool_calls"]:
                func_name = tool_call["function"]["name"]
                func_args = tool_call["function"]["arguments"]

                if func_name in self.tool_functions:
                    result = self.tool_functions[func_name](**func_args)
                    logger.info(f"工具调用: {func_name}({func_args}) → {result}")
                    # 记录工具调用到 last_context
                    if self.context_holder and hasattr(self.context_holder, 'last_context'):
                        self.context_holder.last_context.setdefault("tool_calls", []).append({
                            "name": func_name,
                            "arguments": func_args,
                            "result": result
                        })
                    # 添加工具结果作为新消息
                    messages.append({
                        "role": "tool",
                        "content": str(result),
                        "name": func_name
                    })
                else:
                    logger.warning(f"未找到工具: {func_name}")
                    messages.append({
                        "role": "tool",
                        "content": f"错误：未找到工具 {func_name}",
                        "name": func_name
                    })

            # 再次调用模型，获取下一轮响应
            try:
                second_response = self.llm_engine.chat(messages)
                current_message = second_response.get("message", {})
            except Exception as e:
                logger.error(f"多轮工具调用失败: {e}")
                break

        # 返回最终文本
        final_text = current_message.get("content", "")
        if not final_text:
            final_text = "（处理完成）"
        return final_text, messages

