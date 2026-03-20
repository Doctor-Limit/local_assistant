# assistant.py
from . import tools
from .config import AssistantConfig
from .memory import MemoryManager, MemoryItem
from .state import State
from .rules import RuleEngine
from .llm import LLMEngine
from .response import ResponseGenerator
from .retriever import Retriever
import time
import inspect
import os
import re

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
        if self.config.rag_enable:
            self.retriever = Retriever()
            if os.path.exists(self.config.knowledge_file):
                self.retriever.load_from_file(self.config.knowledge_file)
            #     # 打印加载的块数和第一个块样本
            #     if hasattr(self.retriever, 'collection'):
            #         count = self.retriever.collection.count()
            #         print(f"知识库加载成功，共 {count} 个块")
            #     else:
            #         print("知识库加载完成，但无法获取计数")
            # else:
            #     print(f"警告：知识库文件 {self.config.knowledge_file} 不存在")

        self._init_role()

        # 定义工具函数列表
        self.tool_functions = [
            tools.get_current_time,
            tools.get_weather,
            tools.calculate,
            tools.execute_safe_command
        ]
        # 将工具函数转换为工具描述字典（供Ollama使用）
        self.tools_def = [self._function_to_tool(func) for func in self.tool_functions]

        # 初始化 ResponseGenerator，传入工具函数和 LLM 引擎
        self.response_gen = ResponseGenerator(
            self.config,
            tool_functions=self.tool_functions,
            llm_engine=self.llm
        )



    def _function_to_tool(self, func):
        """将Python函数转换为OpenAI格式的工具描述"""
        signature = inspect.signature(func)
        docstring = inspect.getdoc(func) or ""
        properties = {}
        required = []
        for name, param in signature.parameters.items():
            # 简单假设所有参数都是字符串（可扩展类型推断）
            properties[name] = {"type": "string", "description": ""}
            if param.default == inspect.Parameter.empty:
                required.append(name)
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

    def _handle_direct_tools(self, text: str) -> str or None:
        """
        直接匹配特定意图并调用工具，返回工具结果字符串，若不匹配则返回 None。
        """
        # 时间查询：匹配“时间”、“几点”、“现在时间”
        if re.search(r"时间|几点|现在.*时间", text, re.IGNORECASE):
            return tools.get_current_time()

        # 天气查询：匹配“XX天气”，提取城市名（2-5个中文字符）
        match = re.search(r"([\u4e00-\u9fa5]{2,5})天气", text)
        if match:
            city = match.group(1)
            return tools.get_weather(city)

        # 简单计算：包含数字和运算符，且可能含有“计算”等词
        if re.search(r"[0-9\+\-\*/\(\)]+", text) and re.search(r"计算|等于|多少", text, re.IGNORECASE):
            # 提取纯数学表达式（过滤掉所有非数字、运算符和括号）
            expr = re.sub(r"[^0-9\+\-\*/\(\)]", "", text)
            if expr:
                result = tools.calculate(expr)
                return f"计算结果：{result}"

        # 未匹配
        return None

    def _init_role(self):
        """角色初始化：将系统角色设定存入状态"""
        self.system_role = self.config.role
        self.state.set("initialized", True)

    def process_stream(self, user_input: str):
        t0 = time.time()

        # 1. 记录用户输入
        self.memory.add(MemoryItem("user", user_input, time.time()))
        self.state.increment("turn_count")
        self.state.set("last_input_time", time.time())

        # 2. 规则引擎快速响应（流式）
        if self.rules:
            rule_response = self.rules.check(user_input)
            if rule_response:
                yield rule_response
                self.memory.add(MemoryItem("assistant", rule_response, time.time()))
                self.state.increment("turn_count")
                self.state.set("last_input_time", time.time())
                print(f"规则引擎总耗时：{time.time() - t0:.2f}s")
                return

        # 3. 直接工具意图快速响应（流式）
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
        messages = [{"role": "system", "content": self.system_role}]

        # 获取历史记忆（排除当前用户输入）
        recent = self.memory.get_recent()
        if recent and recent[-1].role == "user" and recent[-1].content == user_input:
            history = recent[:-1]  # 去掉最后一条（当前用户输入）
        else:
            history = recent
        history = history[-self.config.max_history:]  # 限制条数
        for item in history:
            messages.append({"role": item.role, "content": item.content})

        # 5. RAG 检索
        if self.retriever:
            t_rag = time.time()
            related_docs = self.retriever.search(user_input, top_k=self.config.rag_top_k)
            print(f"RAG检索耗时: {time.time() - t_rag:.2f}s")
            if related_docs:
                knowledge = "\n\n".join(related_docs)
                user_content = (
                    f"请严格基于以下知识回答，只使用知识中的原话，不要添加解释：\n\n"
                    f"【知识】\n{knowledge}\n\n【问题】\n{user_input}"
                )
            else:
                user_content = user_input
        else:
            user_content = user_input

        # 添加当前用户消息（可能携带知识）
        messages.append({"role": "user", "content": user_content})

        t2 = time.time()
        # 6. 调用流式 LLM，边收集边输出
        full_response = ""
        try:
            for chunk in self.llm.chat_stream(messages, tools=self.tools_def):
                full_response += chunk
                yield chunk
        except Exception as e:
            error_msg = f"流式生成出错: {e}"
            print(error_msg)
            yield error_msg
            full_response = error_msg

        t3 = time.time()
        print(f"LLM调用耗时：{t3 - t2:.2f}s")

        # 7. 记录助手回复到记忆
        self.memory.add(MemoryItem("assistant", full_response, time.time()))

        # 8. 更新状态（可根据 full_response 内容调整情绪等）
        self.state.increment("turn_count")
        self.state.set("last_input_time", time.time())
        if "开心" in full_response or "高兴" in full_response:
            self.state.set("mood", "happy")
        else:
            self.state.set("mood", "neutral")

        t4 = time.time()
        print(f"总耗时：{t4 - t0:.2f}s")
        # 生成器自然结束，无需 return


    def reset(self):
        """重置对话（清空记忆和状态）"""
        self.memory.clear()
        self.state = State(state_file=self.config.state_file)
        self._init_role()
