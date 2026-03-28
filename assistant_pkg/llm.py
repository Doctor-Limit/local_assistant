import requests
import json
import logging
from typing import List, Dict, Optional, Any, Generator, Iterable

logger = logging.getLogger(__name__)

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

    # 非流式方法
    def chat(self, messages: List[Dict[str, str]], tools: Optional[List[dict]] = None, timeout: int = 30) -> Dict[
        str, Any]:
        if self.use_cloud:
            return self._chat_cloud(messages, tools, timeout)
        else:
            return self._chat_ollama(messages, tools, timeout)

    def _chat_ollama(self, messages, tools, timeout):
        logger.debug(f"调用 Ollama，消息数量: {len(messages)}")
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "num_ctx": self.config.llm_num_ctx,
            "stream": False
        }
        if tools:
            payload["tools"] = tools
        try:
            resp = requests.post(self.ollama_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"非流式调用失败：{e}")
            return {"message": {"content": "调用失败"}}

    def _stream_response(self, lines_generator: Generator[bytes, None, None]) -> Generator[str, None, None]:
        """统一处理 SSE 流式响应，返回解析后的文本块"""
        for line in lines_generator:
            if not line:
                continue
            line = line.decode('utf-8').strip()
            if line.startswith('data:'):
                line = line[5:].strip()
            if line == '[DONE]':
                break
            if not line:
                continue
            try:
                data = json.loads(line)
                # Ollama 格式
                content = data.get('message', {}).get('content', '')
                if content:
                    yield content
                # OpenAI 兼容格式（用于云端）
                elif data.get('choices'):
                    delta = data.get('choices', [{}])[0].get('delta', {})
                    content = delta.get('content', '')
                    if content:
                        yield content
            except json.JSONDecodeError:
                continue

    def _chat_cloud(self, messages, tools, timeout):
        headers = {
            "Authorization": f"Bearer {self.cloud_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.cloud_model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False
        }
        try:
            resp = requests.post(self.cloud_url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            # 将openAI格式提取内容并转换为 Ollama 格式，否则当前云端模型返回的json为空
            content = data["choices"][0]["message"]["content"]
            return {"message": {"content": content}}
        except Exception as e:
            print(f"云端非流式调用失败：{e}")
            return {"message": {"content": "调用失败"}}

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
            # 设置连接超时5秒，读取超时30秒
            with requests.post(self.ollama_url, json=payload, stream=True, timeout=(5, 30)) as resp:
                resp.raise_for_status()
                yield from self._stream_response(resp.iter_lines())
        except requests.Timeout:
            print("流式调用超时")
            yield "请求超时，请稍后重试。"
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
        # 注意：工具调用格式需根据云端 API 适配
        # 若云端 API 支持函数调用，可按其文档补充
        try:
            with requests.post(self.cloud_url, json=payload, headers=headers, stream=True, timeout=(10, 120)) as resp:
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
        except requests.Timeout:
            print("云端流式调用超时")
            yield "云端请求超时，请稍后重试。"
        except Exception as e:
            print(f"云端流式调用失败：{e}")
            yield "抱歉，云端服务出错了。"
