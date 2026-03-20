# llm.py
import requests
import json
from typing import List, Dict, Optional, Any

class LLMEngine:
    """封装Ollama API调用，支持chat接口和工具调用"""
    def __init__(self, config):
        self.config = config
        self.chat_url = f"{config.ollama_url.rstrip('/')}/api/chat"

    def chat_stream(self, messages: List[Dict[str, str]], tools: Optional[List[dict]] = None):
        """
        流式调用Ollama的chat接口，逐块返回内容
        """
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            # 修正参数名：max_token -> max_tokens（Ollama API 使用 max_tokens）
            "max_tokens": self.config.max_tokens,
            "num_ctx": self.config.llm_num_ctx,
            "stream": True
        }
        if tools:
            payload["tools"] = tools

        try:
            with requests.post(self.chat_url, json=payload, stream=True) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        # 解析行数据
                        line = line.decode('utf-8').strip()
                        # Ollama 的流式响应每行是一个 JSON 对象，没有 "data:" 前缀
                        # 如果 line 以 "data:" 开头（某些API可能），则去掉前缀，否则直接解析
                        if line.startswith('data:'):
                            line = line[5:].strip()
                        if line == '[DONE]':
                            break
                        if line:
                            try:
                                data = json.loads(line)
                                # 从响应中提取内容增量
                                # Ollama 的响应格式：{"model":"...","created_at":"...","message":{"role":"assistant","content":"..."},"done":false}
                                content = data.get('message', {}).get('content', '')
                                if content:
                                    yield content
                            except json.JSONDecodeError as e:
                                print(f"JSON解析错误: {e}, line: {line}")
        except Exception as e:
            print(f"流式调用失败：{e}")
            yield "抱歉，出错了。"


    def generate(self, prompt: str, system: str = None, context: List[Dict] = None) -> str:
        """
        为了向后兼容，保留generate方法，内部调用chat
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": prompt})
        response = self.chat(messages)
        content = response.get("message", {}).get("content", "")
        return content.encode('utf-8', 'ignore').decode('utf-8')

