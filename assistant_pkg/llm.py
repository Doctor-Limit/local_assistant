import requests
import json
from typing import List, Dict, Optional, Any

class LLMEngine:
    def __init__(self, config):
        self.config = config
        self.ollama_url = f"{config.ollama_url.rstrip('/')}/api/chat"
        self.use_cloud = getattr(config, 'use_cloud_api', False)
        self.cloud_url = getattr(config, 'cloud_base_url', '')
        self.cloud_api_key = getattr(config, 'cloud_api_key', '')
        self.cloud_model = getattr(config, 'cloud_model', 'Qwen/Qwen2.5-7B-Instruct')

    def chat_stream(self, messages: List[Dict[str, str]], tools: Optional[List[dict]] = None):
        if self.use_cloud:
            return self._chat_stream_cloud(messages, tools)
        else:
            return self._chat_stream_ollama(messages, tools)

    def _chat_stream_ollama(self, messages, tools):
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
            with requests.post(self.ollama_url, json=payload, stream=True) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data:'):
                            line = line[5:].strip()
                        if line == '[DONE]':
                            break
                        if line:
                            try:
                                data = json.loads(line)
                                content = data.get('message', {}).get('content', '')
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                pass
        except Exception as e:
            print(f"流式调用失败：{e}")
            yield "抱歉，出错了。"

    def _chat_stream_cloud(self, messages, tools):
        """调用云端 API（OpenAI 兼容格式）"""
        headers = {
            "Authorization": f"Bearer {self.cloud_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.cloud_model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True
        }
        # 注意：工具调用格式需根据云端 API 适配，这里先不传 tools 保持简洁
        # 如果你的云端 API 支持函数调用，可按其文档补充
        try:
            with requests.post(self.cloud_url, json=payload, headers=headers, stream=True) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data:'):
                            line = line[5:].strip()
                        if line == '[DONE]':
                            break
                        if line:
                            try:
                                data = json.loads(line)
                                delta = data.get('choices', [{}])[0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                pass
        except Exception as e:
            print(f"云端流式调用失败：{e}")
            yield "抱歉，云端服务出错了。"
