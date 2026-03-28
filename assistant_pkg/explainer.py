# explainer.py
import json
import time
import os
import re
from typing import Dict, Optional, TypedDict
from .cache import cache

class ExplanationData(TypedDict):
    text: str
    mermaid: Optional[str]
    citations: list

class ExplanationGenerator:
    def __init__(self, llm_engine, template_file="explanation_templates.json", few_shot_file="few_shot_examples.json"):
        self.llm_engine = llm_engine
        self.templates = self._load_templates(template_file)
        self.few_shots = self._load_few_shots(few_shot_file)
        self.cache = cache  # 将全局缓存实例赋值给实例属性

    def _load_templates(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"模板文件 {filepath} 不存在，使用内置默认模板")
            return self._get_default_templates()
        except Exception as e:
            print(f"加载模板文件失败: {e}，使用内置默认模板")
            return self._get_default_templates()

    def _load_few_shots(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"few-shot 文件 {filepath} 不存在，使用空示例库")
            return {}
        except Exception as e:
            print(f"加载 few-shot 文件失败: {e}，使用空示例库")
            return {}

    #决策树方法
    def build_symbolic_tree(self, original_answer: str, context: Dict) -> Dict:
        """
        根据上下文构建符号推理树。
        返回格式：
        {
            "root": "最终答案",
            "children": [
                {"text": "...", "type": "knowledge", "source": "citation_0"},
                {"text": "...", "type": "reasoning"},
                {"text": "...", "type": "tool", "details": {...}}
            ]
        }
        """
        tree = {
            "root": original_answer,
            "children": []
        }

        # 添加检索到的知识
        citations = context.get("citations", [])
        for i, doc in enumerate(citations):
            tree["children"].append({
                "text": f"📚 知识库片段 {i + 1}: {doc[:200]}...",
                "type": "knowledge",
                "source": f"citation_{i}"
            })

        # 添加工具调用
        tool_calls = context.get("tool_calls", [])
        for tc in tool_calls:
            tool_text = f"🔧 调用工具 {tc['name']}({tc['arguments']}) → {tc['result']}"
            tree["children"].append({
                "text": tool_text,
                "type": "tool",
                "details": tc
            })

        # 添加推理步骤（如果没有显式推理，则从原始答案中提取，或由LLM生成）
        # 此处简单处理：若没有显式推理，则默认添加一条“基于以上信息推导出答案”
        if not any(c["type"] == "reasoning" for c in tree["children"]):
            tree["children"].append({
                "text": "💡 推理：综合以上信息，得出答案。",
                "type": "reasoning"
            })

        return tree

    def generate_reasoning_chain(self, original_answer: str, context: Dict) -> Dict:
        """
        调用 LLM 生成结构化的推理链（JSON格式）。
        返回格式示例：
        {
            "root": "最终答案",
            "children": [
                {"text": "根据知识库：...", "type": "knowledge"},
                {"text": "推理：因为...所以...", "type": "reasoning"},
                {"text": "工具调用：...", "type": "tool"}
            ]
        }
        """
        # 构建上下文信息
        citations = context.get("citations", [])
        tool_calls = context.get("tool_calls", [])
        relevant_memories = context.get("relevant_memories", [])
        user_input = context.get("user_input", "")

        context_str = ""
        if citations:
            context_str += "知识库片段：\n" + "\n".join(citations) + "\n"
        if tool_calls:
            context_str += "工具调用记录：\n" + "\n".join(
                [f"{tc['name']}({tc['arguments']}) → {tc['result']}" for tc in tool_calls]) + "\n"
        if relevant_memories:
            context_str += "相关历史记忆：\n" + "\n".join([f"{mem.content}" for mem in relevant_memories]) + "\n"

        prompt = f"""
    请基于以下信息，为我的回答生成一个结构化的推理链（JSON格式），展示得出最终答案的步骤。
    回答：{original_answer}
    上下文：
    {context_str}

    请返回 JSON，格式如下：
    {{
        "root": "最终答案（可以引用回答中的关键点）",
        "children": [
            {{"text": "步骤1描述", "type": "knowledge/reasoning/tool"}},
            {{"text": "步骤2描述", "type": "..."}}
        ]
    }}
    注意：type 可以是 "knowledge"（知识库）、"tool"（工具调用）、"reasoning"（推理）等。每个步骤要简洁明了。
    """
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_engine.chat(messages, timeout=self.llm_engine.config.llm_timeout)
            content = response.get("message", {}).get("content", "")
            data = self._parse_json(content)
            if data and isinstance(data, dict):
                return data
            else:
                return {"root": original_answer, "children": []}
        except Exception as e:
            print(f"生成推理链失败: {e}")
            return {"root": original_answer, "children": []}

    def tree_to_mermaid(self, tree: Dict) -> str:
        """将符号树转换为 mermaid 流程图（graph TD）"""
        if not tree or not tree.get("children"):
            return ""
        lines = ["graph TD"]
        node_id = 0
        root_node = f"root{node_id}"
        root_text = tree.get("root", "答案")[:50]
        lines.append(f"    {root_node}[{root_text}]")
        node_id += 1

        for child in tree["children"]:
            child_id = f"node{node_id}"
            lines.append(f"    {child_id}[{child['text'][:50]}]")
            lines.append(f"    {root_node} --> {child_id}")
            node_id += 1
        return "\n".join(lines)

    def _get_default_templates(self):
        """内置默认模板（仅包含 Ti 示例，其他功能可简化）"""
        return {
            "Ti": {
                "description": "内倾思考 - 注重逻辑链条、因果推理和内部一致性",
                "structure": [
                    "1. **关键前提**：列出回答依赖的核心事实或假设。",
                    "2. **推理步骤**：用编号列出从前提推导出结论的逻辑链（每步注明依据）。",
                    "3. **内部一致性检查**：说明该推理链与之前记忆或知识库中的信息是否矛盾。",
                    "4. **结论**：总结最终答案的逻辑必然性。"
                ],
                "guidelines": "每一步推理必须使用“因为…所以…”或“如果…则…”的句式。每个推理步骤必须明确依据（如“根据知识库”、“根据常识”、“根据工具结果”）。"
            },
            # 可根据需要添加其他功能的默认模板，或仅保留 Ti 让其他功能回退到 Ti
        }

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

    def _build_prompt(self, function: str, original_answer: str, context: Dict = None) -> str:
        template = self.templates.get(function)
        if not template:
            raise ValueError(f"未找到功能 {function} 的模板")

        structure = "\n".join(template["structure"])
        guidelines = template.get("guidelines", "")
        guidelines_text = f"\n【额外要求】\n{guidelines}\n" if guidelines else ""

        # 获取 few-shot 示例
        few_shot_data = self.few_shots.get(function, {})
        main_example = few_shot_data.get("main", "")
        main_example_text = f"\n【主示例】\n{main_example}\n" if main_example else ""
        few_shot_examples = few_shot_data.get("few_shot", [])
        few_shot_text = ""
        if few_shot_examples:
            few_shot_text = "【更多示例】\n"
            for ex in few_shot_examples:
                few_shot_text += f"问题：{ex['question']}\n回答：{ex['answer']}\n解释：\n{ex['explanation']}\n\n"

        # 构建上下文描述（同之前）
        context_str = ""
        if context:
            if context.get("citations"):
                context_str += "\n\n检索到的知识片段：\n" + "\n".join(context["citations"])
            if context.get("relevant_memories"):
                mem_str = "\n".join([f"- [{item.source_type}] {item.content}" for item in context["relevant_memories"]])
                context_str += f"\n\n相关历史记忆：\n{mem_str}"
            if context.get("tool_calls"):
                context_str += "\n\n工具调用记录：\n" + "\n".join(
                    f"- {tc['name']}({tc['arguments']}) → {tc['result']}" for tc in context["tool_calls"]
                )
            if context and context.get("history_explanations"):
                hist_str = "\n".join([f"第 {h['turn']} 轮: {h['summary']}" for h in context["history_explanations"]])
                context_str += f"\n\n之前已给出的解释摘要：\n{hist_str}\n请在生成新解释时参考以上内容，保持逻辑一致，避免矛盾。"
            if context.get("user_profile_confidence"):
                func, conf = list(context["user_profile_confidence"].items())[0]
                context_str += f"\n\n用户画像显示：对 {func} 功能的置信度为 {conf}，因此优先采用此风格。"
            if context.get("internal_state"):
                state = context["internal_state"]
                context_str += f"\n\n当前内部状态：情绪={state.get('mood')}，对话轮次={state.get('turn_count')}"
                if state.get("rule_triggered"):
                    context_str += "，规则引擎已触发"
                if state.get("rag_used"):
                    context_str += "，已使用RAG检索"
                if state.get("tool_used"):
                    context_str += "，已使用工具"

        prompt = f"""你是一个助手，需要为用户解释你的回答。请严格遵循以下结构生成解释，并参考示例格式。**请以JSON格式返回**，包含以下字段：
    - "text": 解释文本，按照结构输出。
    - "mermaid": 可选的流程图代码（仅当功能为Ti且适合流程图时提供，否则为null）。
    - "citations": 引用来源列表，每个元素包含 "content" (引用的原文) 和 "type" ("knowledge"/"tool"/"reasoning")。

    【认知功能】：{function} ({template["description"]})

    【输出结构】：
    {structure}
    {guidelines_text}

    {main_example_text}
    {few_shot_text}

    【原始回答】：
    {original_answer}
    {context_str}

    请生成JSON："""
        return prompt

    def explain(self, function: str, original_answer: str, context: Optional[Dict] = None,
                use_cache: bool = True) -> ExplanationData:
        # 生成缓存键
        question = context.get("user_input", "") if context else ""
        profile = context.get("user_profile_confidence", {})
        cache_key = self.cache.get_key(question, profile, function) if use_cache else None
        cached = self.cache.get(cache_key) if use_cache else None
        if cached:
            print("使用缓存的解释")
            return cached

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                prompt = self._build_prompt(function, original_answer, context)
                messages = [{"role": "user", "content": prompt}]
                response = self.llm_engine.chat(messages, timeout=self.llm_engine.config.llm_timeout)
                content = response.get("message", {}).get("content", "")
                data = self._parse_json(content)
                if data and isinstance(data, dict):
                    result = {
                        "text": data.get("text", self._fallback_explanation(function, original_answer, context)),
                        "mermaid": data.get("mermaid"),
                        "citations": data.get("citations", [])
                    }
                    if use_cache:
                        self.cache.set(cache_key, result)
                    return result
            except Exception as e:
                print(f"解释生成尝试 {attempt + 1}/{max_retries + 1} 失败：{e}")
                if attempt < max_retries:
                    time.sleep(0.5)
                    continue
                # 最后一次尝试失败，跳出循环进入降级
                break

        # 降级解释（不缓存）
        return {
            "text": self._fallback_explanation(function, original_answer, context),
            "mermaid": None,
            "citations": []
        }

    def _fallback_explanation(self, function: str, original_answer: str, context: Dict = None) -> str:
        """
        当 LLM 调用失败时，使用规则生成简单的解释。
        """
        # 根据是否有 citations 和 tool_calls 构造简单解释
        parts = []
        if context and context.get("citations"):
            parts.append(f"根据知识库中的 {len(context['citations'])} 条相关内容：")
            for idx, doc in enumerate(context["citations"][:2]):  # 只展示前两条
                parts.append(f"  - {doc[:100]}...")
        if context and context.get("tool_calls"):
            parts.append(f"使用了工具调用：{', '.join([tc['name'] for tc in context['tool_calls']])}")
        if not parts:
            parts.append("基于我的内部知识和对话历史回答。")
        parts.append("（由于解释生成服务暂时不可用，以上是简化版说明。）")
        return "\n".join(parts)

    # 快捷方法保持不变
    def explain_by_ti(self, answer: str, context: dict) -> str:
        return self.explain("Ti", answer, context)

    def explain_by_ti(self, answer: str, context: dict) -> str:
        return self.explain("Ti", answer, context)

    def explain_by_te(self, answer: str, context: dict) -> str:
        return self.explain("Te", answer, context)

    def explain_by_fi(self, answer: str, context: dict) -> str:
        return self.explain("Fi", answer, context)

    def explain_by_fe(self, answer: str, context: dict) -> str:
        return self.explain("Fe", answer, context)

    def explain_by_si(self, answer: str, context: dict) -> str:
        return self.explain("Si", answer, context)

    def explain_by_se(self, answer: str, context: dict) -> str:
        return self.explain("Se", answer, context)

    def explain_by_ni(self, answer: str, context: dict) -> str:
        return self.explain("Ni", answer, context)

    def explain_by_ne(self, answer: str, context: dict) -> str:
        return self.explain("Ne", answer, context)