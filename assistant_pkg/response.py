# response.py
from typing import List, Dict, Any, Tuple

class ResponseGenerator:
    def __init__(self, config, tool_functions=None, llm_engine=None):
        self.config = config
        self.tool_functions = {func.__name__: func for func in (tool_functions or [])}
        self.llm_engine = llm_engine  # 用于二次调用

    def generate(self, llm_response: dict, messages: List[dict]) -> Tuple[str, List[dict]]:
        """
        处理 LLM 响应，如果包含工具调用则执行并返回结果，否则返回文本
        :param llm_response: 从 LLMEngine.chat 返回的完整响应
        :param messages: 当前消息列表（会被修改）
        :return: (最终回复文本, 更新后的消息列表)
        """
        message = llm_response.get("message", {})
        if "tool_calls" in message and message["tool_calls"]:
            # 处理每个工具调用
            for tool_call in message["tool_calls"]:
                func_name = tool_call["function"]["name"]
                func_args = tool_call["function"]["arguments"]
                if func_name in self.tool_functions:
                    result = self.tool_functions[func_name](**func_args)
                    print(f"工具 {func_name} 返回：{result}")
                    messages.append({
                        "role": "tool",
                        "content": str(result),
                        "name": func_name
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "content": f"错误：未找到工具 {func_name}",
                        "name": func_name
                    })
            # 重新调用模型获取最终回复
            second_response = self.llm_engine.chat(messages)
            final_text = second_response.get("message", {}).get("content", "")
            return final_text, messages
        else:
            # 普通回复
            return message.get("content", ""), messages

