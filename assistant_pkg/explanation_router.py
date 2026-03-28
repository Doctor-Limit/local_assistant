# 解释路由与可视化准备
from typing import Dict, Optional, Tuple, List, TypedDict
from .user_profile import UserProfile
from .explainer import ExplanationGenerator

class VizData(TypedDict):
    citations: List[str]
    confidence: float
    reasoning_chain: str
    tool_calls: List[Dict]
    mermaid: Optional[str]
    citations_used: List[Dict]
    all_functions_confidence: Dict[str, float]

class ExplanationRouter:
    """
    根据任务类型和用户偏好，选择合适的解释器，并返回解释文本及可视化数据。
    """
    def __init__(self, user_profile: UserProfile, explainer: ExplanationGenerator):
        self.user_profile = user_profile
        self.explainer = explainer

    #将符号解释与自然语言解释集合
    def route(self, task_type: str, original_answer: str,
              context: Optional[Dict] = None,
              history_explanations: Optional[List[Dict]] = None) -> Tuple[str, Dict]:
        """
        生成解释，并返回解释文本和可视化数据。
        :param task_type: 任务类型（如 "knowledge"）
        :param original_answer: 原始回答
        :param context: 上下文信息（citations, tool_calls, relevant_memories 等）
        :param history_explanations: 之前几轮的解释摘要列表，每个元素包含 "turn", "summary", "full_text"
        """
        # 获取当前用户偏好的认知功能及其置信度
        func, confidence = self.user_profile.get_preferred_function(task_type)
        if context is not None:
            context["user_profile_confidence"] = {func: confidence}
            # 将历史解释摘要注入 context，供 explainer 使用
            if history_explanations:
                context["history_explanations"] = history_explanations

        # 生成自然语言解释（一次调用）
        explanation_data = self.explainer.explain(func, original_answer, context)
        explanation_text = explanation_data.get("text", "")
        if not explanation_text:
            explanation_text = "（无法生成解释，但您可以继续提问。）"

        # 生成推理链（用于可视化）
        reasoning_chain = self.explainer.generate_reasoning_chain(original_answer, context or {})
        mermaid_reasoning = self.explainer.tree_to_mermaid(reasoning_chain)

        # 构建可视化数据
        viz_data = {
            "citations": context.get("citations", []) if context else [],
            "confidence": confidence,
            "reasoning_chain": explanation_text,
            "tool_calls": context.get("tool_calls", []) if context else [],
            "mermaid": explanation_data.get("mermaid"),
            "citations_used": explanation_data.get("citations", []),
            "all_functions_confidence": self.user_profile.data.get("cognitive_functions", {}),
            "reasoning_chain_json": reasoning_chain,
            "mermaid_reasoning": mermaid_reasoning
        }
        return explanation_text, viz_data