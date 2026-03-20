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
                        line = line.decode('utf-8').strip()
                        # Ollama 的流式响应每行是一个 JSON 对象，没有 "data:" 前缀
                        if line.startswith('data:'):
                            line = line[5:].strip()
                        if line == '[DONE]':
                            break
                        if line:
                            try:
                                data = json.loads(line)
                                # Ollama 的响应格式：{"model":"...","created_at":"...","message":{"role":"assistant","content":"..."},"done":false}
                                content = data.get('message', {}).get('content', '')
                                if content:
                                    yield content
                            except json.JSONDecodeError as e:
                                print(f"JSON解析错误: {e}, line: {line}")
        except Exception as e:
            print(f"流式调用失败：{e}")
            yield "抱歉，出错了。"


