# assistant.py
import time
import inspect
import os
import re
import json
import logging
from typing import List, Dict, Optional, Generator
from . import tools
from .config import AssistantConfig
from .memory import MemoryManager, MemoryItem
from .state import State
from .rules import RuleEngine
from .llm import LLMEngine
from .response import ResponseGenerator
from .retriever import Retriever
from .user_profile import UserProfile  # 导入用户画像，用于获取置信度
from .answer_explanation_generator import AnswerExplanationGenerator

logger = logging.getLogger(__name__)

class Assistant:
    """助手主控制器，整合所有模块"""
    def __init__(self, config_path=None):
        self.config = AssistantConfig(config_path)
        self.memory = MemoryManager(max_size=self.config.memory_size,
                                    memory_file=self.config.memory_file)
        self.state = State(state_file=self.config.state_file)
        self.rules = RuleEngine() if self.config.enable_rules else None
        self.llm = LLMEngine(self.config)
        self.retriever = None
        self.last_context = {}  # 存储最近一次对话的上下文（包括缓存的解释）
        self.user_profile = UserProfile()  # 用于获取认知功能置信度

        # 自动获取所有注册的工具,不需要构建工具列表
        self.tool_functions = tools.get_registered_tools()
        self.tools_def = [self._function_to_tool(func) for func in self.tool_functions]
        self.tool_examples = self._load_tool_examples()

        # 加载风格示例
        self.style_examples = self._load_style_examples()

        if self.config.rag_enable:
            self.retriever = Retriever()
            if os.path.exists(self.config.knowledge_file):
                self.retriever.load_from_file(self.config.knowledge_file)

        self._init_role()

        # 初始化 ResponseGenerator，传入工具函数和 LLM 引擎
        self.response_gen = ResponseGenerator(
            self.config,
            tool_functions=self.tool_functions,
            llm_engine=self.llm,
            context_holder=self  # 传入 self，以便访问 last_context
        )

        self.answer_gen = AnswerExplanationGenerator(
            self.llm,
            self.user_profile,
            timeout=self.config.llm_timeout  # 从配置中读取
        )

    def _function_to_tool(self, func):
        signature = inspect.signature(func)
        docstring = inspect.getdoc(func) or ""
        properties = {}
        required = []
        for name, param in signature.parameters.items():
            # 根据参数注解推断类型
            annotation = param.annotation
            if annotation == int:
                param_type = "integer"
            elif annotation == float:
                param_type = "number"
            elif annotation == bool:
                param_type = "boolean"
            else:
                param_type = "string"
            properties[name] = {"type": param_type, "description": ""}
            if param.default == inspect.Parameter.empty:
                required.append(name)
            description = docstring
            if func.__name__ == "get_weather":
                description += " 例如：用户说‘北京天气如何？’时调用此工具。"
            elif func.__name__ == "calculate":
                description += " 例如：用户说‘计算 12+34*5’时调用此工具。"
            elif func.__name__ == "execute_safe_command":
                description += " 例如：用户说‘安全执行命令’时调用此工具。"
            elif func.__name__ == "get_current_time":
                description += " 例如：用户说‘现在几点了’时调用此工具。"
        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": docstring,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

    def _load_tool_examples(self):
        filepath = os.path.join(os.path.dirname(__file__), "..", "tool_examples.json")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("examples", [])
        return []

    #回答风格类型
    def _load_style_examples(self):
        filepath = os.path.join(os.path.dirname(__file__), "..", "style_examples.json")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    # 场景匹配方法
    def _match_scenario(self, user_input: str) -> Optional[str]:
        """根据用户输入匹配场景关键词，返回场景ID或None"""
        keywords_map = {
            "trolley_problem": ["电车", "伦理困境", "拉杆", "轨道", "5个人", "1个人"],
            "job_offer": ["offer", "工作", "加班", "工资", "平衡"],
            "rock_paper_scissors": ["石头剪刀布", "游戏策略", "石头", "剪刀", "布"]
        }
        user_lower = user_input.lower()
        for scene, keywords in keywords_map.items():
            if any(kw in user_lower for kw in keywords):
                return scene
        return None

    def _handle_direct_tools(self, text: str) -> str or None:
        """
        直接匹配特定意图并调用工具，返回工具结果字符串，若不匹配则返回 None。
        """
        if re.search(r"时间|几点|现在.*时间", text, re.IGNORECASE):
            return tools.get_current_time()
        match = re.search(r"([\u4e00-\u9fa5]{2,5})天气", text)
        if match:
            city = match.group(1)
            return tools.get_weather(city)
        if re.search(r"[0-9\+\-\*/\(\)]+", text) and re.search(r"计算|等于|多少", text, re.IGNORECASE):
            expr = re.sub(r"[^0-9\+\-\*/\(\)]", "", text)
            if expr:
                result = tools.calculate(expr)
                return f"计算结果：{result}"
        return None

    def _intent_direct_tools(self, text: str) -> str or None:
        """
        意图识别前置：直接处理常见工具意图，返回结果字符串，若不匹配则返回 None。
        此方法比 _handle_direct_tools 更宽泛，用于复杂语句的快速响应。
        """
        # 时间查询
        if re.search(r"时间|几点|现在.*时间", text, re.IGNORECASE):
            return tools.get_current_time()

        # 天气查询：提取城市名（2-5个中文字符）
        match = re.search(r"([\u4e00-\u9fa5]{2,5})天气", text)
        if match:
            city = match.group(1)
            return tools.get_weather(city)

        # 计算：提取数字和运算符
        if re.search(r"[0-9\+\-\*/\(\)]+", text) and re.search(r"计算|等于|多少", text, re.IGNORECASE):
            expr = re.sub(r"[^0-9\+\-\*/\(\)]", "", text)
            if expr:
                result = tools.calculate(expr)
                return f"计算结果：{result}"

        # 可扩展更多意图
        return None

    def _init_role(self):
        self.system_role = (
                self.config.role +
                " 你可以使用以下工具：时间、天气、计算、执行安全命令。"
                " 如果用户的问题包含多个独立需求，你可以依次调用多个工具来完成。"
                " 例如：用户说“帮我查北京天气，然后计算25*4”，你应该先调用 get_weather(city='北京')，"
                " 然后在得到结果后，再调用 calculate(expression='25*4')。"
        )
        self.state.set("initialized", True)

    def rewrite_query(self, user_input: str) -> str:
        """使用 LLM 改写查询，提高检索精度"""
        prompt = f"请将以下用户问题改写为简洁的关键词或搜索短语，只输出改写结果：\n{user_input}"
        try:
            resp = self.llm.chat([{"role": "user", "content": prompt}], timeout=5)
            rewritten = resp.get("message", {}).get("content", "").strip()
            return rewritten if rewritten else user_input
        except:
            return user_input

    def consolidate_memories(self):
        """将短期记忆中的重要内容转移至长期记忆，并提炼知识"""
        # 选取高置信度的记忆
        high_conf = [item for item in self.memory.short_term if item.confidence > 0.6]
        if not high_conf:
            return
        # 让 LLM 提炼知识
        prompt = "请从以下对话记忆中提炼出有价值的知识点：\n" + "\n".join([item.content for item in high_conf])
        response = self.llm.chat([{"role": "user", "content": prompt}])
        knowledge = response.get("message", {}).get("content", "")
        if knowledge:
            self.memory.add(MemoryItem("system", knowledge, source_type="refined", confidence=0.9))

    def _get_style_name(self, func: str) -> str:
        """返回认知功能的中文名称"""
        styles = {
            "Ti": "逻辑型",
            "Te": "效率型",
            "Fi": "价值观型",
            "Fe": "情感型",
            "Si": "经验型",
            "Se": "行动型",
            "Ni": "洞察型",
            "Ne": "发散型"
        }
        return styles.get(func, "通用型")

    def _get_style_description(self, func: str) -> str:
        """返回风格的详细描述，用于提示模型"""
        descriptions = {
            "Ti": "逻辑严谨，使用“因为…所以…”的句式，注重因果推理和内部一致性",
            "Te": "效率导向，使用编号步骤，引用权威资料，注重结果和可操作性",
            "Fi": "情感共鸣，使用第一人称表达感受，强调个人价值观，温暖鼓励",
            "Fe": "群体和谐，使用“我们”视角，强调社交规范和团队合作",
            "Si": "经验细节，引用过往案例，注重稳定性和可重复性",
            "Se": "当下行动，提供具体操作，描述环境反馈，灵活调整",
            "Ni": "洞察趋势，使用隐喻和抽象概念，预测未来影响",
            "Ne": "发散可能，列举多种方案，跨界联想，开放结局"
        }
        return descriptions.get(func, "清晰、有条理")

    def process_stream(self, user_input: str) -> Generator[str, None, None]:
        t0 = time.time()
        self.last_context = {}

        # 1. 记录用户输入
        self.memory.add(MemoryItem("user", user_input, time.time()))
        self.state.increment("turn_count")
        self.state.set("last_input_time", time.time())

        # 内部状态快照
        self.last_context["internal_state"] = {
            "mood": self.state.get("mood", "neutral"),
            "turn_count": self.state.get("turn_count", 0),
            "rule_triggered": False,
            "rag_used": False,
            "tool_used": False
        }

        # 2. 规则引擎
        if self.rules:
            rule_response = self.rules.check(user_input)
            if rule_response:
                yield rule_response
                self.memory.add(MemoryItem("assistant", rule_response, time.time()))
                self.state.increment("turn_count")
                self.state.set("last_input_time", time.time())
                print(f"规则引擎总耗时：{time.time() - t0:.2f}s")
                return

        # 意图识别前置
        direct_tool_response = self._intent_direct_tools(user_input)
        if direct_tool_response:
            logger.info(f"意图识别命中，直接返回工具结果: {direct_tool_response[:100]}")
            yield direct_tool_response
            self.memory.add(MemoryItem("assistant", direct_tool_response, time.time()))
            self.state.increment("turn_count")
            self.state.set("last_input_time", time.time())
            print(f"意图识别工具总耗时：{time.time() - t0:.2f}s")
            return

        # 3. 直接工具处理（原有快速匹配）
        direct_response = self._handle_direct_tools(user_input)
        if direct_response:
            yield direct_response
            self.memory.add(MemoryItem("assistant", direct_response, time.time()))
            self.state.increment("turn_count")
            self.state.set("last_input_time", time.time())
            print(f"直接工具总耗时：{time.time() - t0:.2f}s")
            return

        t1 = time.time()
        print(f"规则引擎+工具处理耗时：{t1 - t0:.2f}s")

        # 4. 构建消息列表
        # 获取当前偏好的认知功能及置信度（实时）
        current_func, current_conf = self.user_profile.get_preferred_function("knowledge")
        style_desc = self._get_style_description(current_func)
        style_name = self._get_style_name(current_func)

        # 系统提示中加入风格引导和输出格式要求
        system_content = (
            f"{self.system_role} "
            f"请以 **{current_func}** 风格（{style_desc}）回答用户问题。"
            f"\n\n请参考以下示例格式输出答案："
            f"\n- **{current_func}（{style_name}）**：你的回答内容。"
            f"\n注意：回答内容应体现该风格的典型视角，例如 Ti 强调逻辑计算，Fe 关注他人感受，Fi 坚守个人价值观，Ne 提出多种可能。"
            f"\n\n**重要：你只需回答用户当前提出的问题。如果历史对话中已经包含之前的回答，请不要重复回答旧问题。**"
        )
        messages = [{"role": "system", "content": system_content}]

        # 插入风格示例（根据场景匹配）
        if self.style_examples:
            scenes = self.style_examples.get("scenarios", {})
            scene = self._match_scenario(user_input)
            if scene and scene in scenes:
                scene_data = scenes[scene]
                if current_func in scene_data["cognitive_functions"]:
                    func_data = scene_data["cognitive_functions"][current_func]
                    # 插入示例
                    messages.append({"role": "user", "content": scene_data["user_prompt"]})
                    messages.append({"role": "assistant", "content": func_data["response"]})

        # 插入 few-shot 示例
        if getattr(self.config, 'use_few_shot_examples', True) and self.tool_examples:
            for ex in self.tool_examples:
                messages.append({"role": "user", "content": ex["user"]})
                messages.append({"role": "assistant", "content": ex["assistant"]})
            # 加分隔提示
            messages.append({"role": "system", "content": "请参考以上示例，处理接下来的用户请求。"})

        # 添加历史对话
        recent = self.memory.get_recent()
        if recent and recent[-1].role == "user" and recent[-1].content == user_input:
            history = recent[:-1]
        else:
            history = recent
        history = history[-self.config.max_history:]
        for item in history:
            messages.append({"role": item.role, "content": item.content})

        # 5. RAG 检索（含查询改写）
        # 判断是否属于技术类问题，否则进入常规逻辑
        tech_keywords = ["锁", "线程", "synchronized", "HashMap", "JVM", "AQS", "内存", "并发", "数据库", "索引",
                         "事务", "volatile", "ReentrantLock"]
        if self.retriever and any(kw in user_input for kw in tech_keywords):
            t_rag = time.time()
            use_query_rewrite = getattr(self.config, 'query_rewrite', True)
            query_to_search = self.rewrite_query(user_input) if use_query_rewrite else user_input
            related_docs = self.retriever.search(query_to_search, top_k=self.config.rag_top_k)
            print(f"RAG检索耗时: {time.time() - t_rag:.2f}s")
            if related_docs:
                self.last_context["citations"] = related_docs
                knowledge = "\n\n".join(related_docs)
                user_content = (
                    f"请参考以下知识回答用户问题。你可以基于这些知识进行解释、总结或补充，用自己的话清晰地表达。"
                    f"如果知识中没有直接答案，可以根据你的常识回答。\n\n"
                    f"【参考知识】\n{knowledge}\n\n【问题】\n{user_input}"
                )
            else:
                user_content = user_input
        else:
            user_content = user_input

        # 在 RAG 检索之后，添加记忆检索
        if self.memory.long_term:
            relevant_memories = self.memory.search(user_input, top_k=3, include_long_term=True)
            if relevant_memories:
                # 将相关记忆加入到上下文，可以拼接在用户消息之后
                memory_context = "\n\n".join([f"【历史记忆】{item.content}" for item in relevant_memories])
                user_content = f"{user_content}\n\n{memory_context}"
                self.last_context["relevant_memories"] = relevant_memories  # 供解释使用

        messages.append({"role": "user", "content": user_content})

        # 6. 决定是否使用合并模式（一次调用同时获得答案和解释）
        use_merge = getattr(self.config, 'merge_explanation', True)
        if use_merge:
            try:
                answer, explanation_data = self.answer_gen.generate(messages, user_input)
                if answer:
                    # 缓存解释数据
                    self.last_context["explanation"] = explanation_data
                    # 模拟流式输出答案
                    chunk_size = 20
                    for i in range(0, len(answer), chunk_size):
                        yield answer[i:i + chunk_size]
                        time.sleep(0.02)
                    full_response = answer
                else:
                    # 合并调用无结果，回退
                    yield from self._original_stream(messages)
                    return
            except Exception as e:
                print(f"合并调用异常，回退到原逻辑: {e}")
                yield from self._original_stream(messages)
                return
            finally:
                # 每次对话后尝试衰减置信度（内部会检查时间间隔）
                self.user_profile.decay_confidences()
        else:
            yield from self._original_stream(messages)
            return

        t3 = time.time()
        print(f"\nLLM调用耗时：{t3 - t1:.2f}s")

        # 记录最终回复和上下文
        self.last_context["final_answer"] = full_response
        self.memory.add(MemoryItem("assistant", full_response, time.time()))

        # 更新状态
        self.state.increment("turn_count")
        self.state.set("last_input_time", time.time())
        if "开心" in full_response or "高兴" in full_response:
            self.state.set("mood", "happy")
        else:
            self.state.set("mood", "neutral")

        t4 = time.time()
        print(f"总耗时：{t4 - t0:.2f}s")

        self.consolidate_memories()

    def _build_merged_prompt(self, messages: List[Dict], user_input: str) -> str:
        """构建要求返回 JSON 的 prompt，同时获得答案和解释"""
        # 提取系统提示
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        # 提取历史（排除当前用户消息）
        history = [m for m in messages if m["role"] != "system" and not (m["role"] == "user" and m["content"] == user_input)]
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history])
        prompt = f"""你是一个助手，需要同时给出最终答案和解释。请以 JSON 格式返回，包含以下字段：
- "answer": 最终回答文本。
- "explanation": 包含 "style" (如 Ti), "text" (解释文本), "mermaid" (可选流程图), "citations" (引用来源列表，每个元素含 "content" 和 "type")。

系统提示：{system_msg}

历史对话：
{history_str}

用户问题：{user_input}

请返回 JSON："""
        return prompt

    def _original_stream(self, messages: List[Dict]) -> Generator[str, None, None]:
        """原有的流式生成逻辑（回退用）"""
        full_response = ""
        full_response_size = 20
        try:
            for chunk in self.llm.chat_stream(messages, tools=self.tools_def):
                full_response += chunk
                if len(full_response) >= full_response_size:
                    yield full_response
                    full_response = ""
            if full_response:
                yield full_response
        except Exception as e:
            if full_response:
                yield full_response
            error_msg = f"流式生成出错: {e}"
            print(error_msg)
            yield error_msg

    def reset(self):
        """重置对话（清空记忆和状态）"""
        self.memory.clear()
        self.state = State(state_file=self.config.state_file)
        self._init_role()