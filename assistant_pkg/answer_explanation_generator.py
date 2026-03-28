# answer_explanation_generator.py
import json
import time
import re
from typing import List, Dict, Optional, Tuple

class AnswerExplanationGenerator:
    """负责同时生成答案与解释，支持重试与校验"""
    def __init__(self, llm_engine, user_profile, timeout = 60, max_retries=2):
        self.llm_engine = llm_engine
        self.user_profile = user_profile
        self.max_retries = max_retries
        self.timeout = timeout

    def _build_prompt(self, messages: List[Dict], user_input: str, strict: bool = False) -> str:
        """构建要求返回 JSON 的 prompt，可切换严格模式"""
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        history = [m for m in messages if m["role"] != "system" and not (m["role"] == "user" and m["content"] == user_input)]
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history])
        base_prompt = f"""你是一个助手，需要同时给出最终答案和解释。请以 JSON 格式返回，包含以下字段：
- "answer": 最终回答文本。
- "explanation": 包含 "style" (如 Ti), "text" (解释文本), "mermaid" (可选流程图), "citations" (引用来源列表，每个元素含 "content" 和 "type")。

系统提示：{system_msg}

历史对话：
{history_str}

用户问题：{user_input}

请返回 JSON："""
        if strict:
            base_prompt += "\n\n**注意：请只输出一个纯 JSON 对象，不要包含任何其他文本（如代码块标记、解释等）。**"
        return base_prompt

    import re
    import json

    def _parse_json(self, content: str) -> Optional[Dict]:
        if not content:
            return None
        # 尝试直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        # 清理代码块标记
        if "```json" in content:
            content = content.split("```json", 1)[1]
        if "```" in content:
            content = content.split("```", 1)[0]
        content = content.strip()
        # 使用 raw_decode 提取第一个 JSON 对象
        decoder = json.JSONDecoder()
        try:
            obj, idx = decoder.raw_decode(content)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        # 回退到正则匹配
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
        return None

    def _validate_data(self, data: Dict) -> bool:
        """校验返回的 JSON 数据是否包含必需字段"""
        if not isinstance(data, dict):
            return False
        if "answer" not in data:
            return False
        if "explanation" not in data:
            return False
        expl = data["explanation"]
        if not isinstance(expl, dict):
            return False
        if "text" not in expl:
            return False
        # style 可选，citations 可选，mermaid 可选
        return True

    def generate(self, messages: List[Dict], user_input: str) -> Tuple[str, Dict]:
        """返回 (answer, explanation_data) 元组，支持重试"""
        for attempt in range(self.max_retries + 1):
            # 第一次用普通模式，重试用严格模式
            strict = (attempt > 0)
            prompt = self._build_prompt(messages, user_input, strict=strict)
            try:
                resp = self.llm_engine.chat([{"role": "user", "content": prompt}], timeout=self.timeout)
                content = resp.get("message", {}).get("content", "")
                data = self._parse_json(content)
                if data and self._validate_data(data):
                    # 获取当前认知功能及置信度
                    func, confidence = self.user_profile.get_preferred_function("knowledge")
                    data["explanation"]["confidence"] = confidence
                    return data["answer"], data["explanation"]
                else:
                    print(f"合并调用返回无效 JSON（尝试 {attempt+1}/{self.max_retries+1}）")
            except Exception as e:
                print(f"合并调用异常：{e}（尝试 {attempt+1}/{self.max_retries+1}）")
            # 重试前稍作等待
            if attempt < self.max_retries:
                time.sleep(0.5)

        # 所有重试失败，返回空
        return "", {}