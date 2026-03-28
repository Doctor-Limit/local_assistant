# user_profile.py
import json
import os
import time
import hashlib
from typing import Optional, Dict, Tuple, List

class UserProfile:
    """管理用户认知偏好（MBTI 类型或各认知功能强度）"""
    def __init__(self, profile_file: str = "user_profile.json", mapping_file: str = "mbti_mapping.json",
                 feedback_log_file: str = "feedback_log.json"):
        self.profile_file = profile_file
        self.mapping_file = mapping_file
        self.feedback_log_file = feedback_log_file
        self.data = self._load()
        if not self.data:
            self.data = {
                "mbti": "INTP",
                "cognitive_functions": self._get_default_functions("INTP"),
                "last_updated": time.time()
            }
            self._save()

    def get_key(self, question: str, user_profile: dict) -> str:
        norm_question = question.strip().lower()
        # 这里 user_profile 可能是 context 中的 user_profile_confidence 或其他，需兼容
        funcs = user_profile.get("cognitive_functions", {}) if isinstance(user_profile, dict) else {}
        func_str = json.dumps({k: round(v, 2) for k, v in sorted(funcs.items())})
        return hashlib.md5((norm_question + func_str).encode()).hexdigest()

    def _get_default_functions(self, mbti: str) -> dict:
        """从 JSON 映射文件加载指定 MBTI 的认知功能分数，若文件不存在则返回内置默认值"""
        mapping = self._load_mapping()
        default = mapping.get(mbti.upper())
        if default:
            return default.copy()
        # 内置默认（INTP）
        return {
            "Ti": 0.8, "Te": 0.2,
            "Fi": 0.3, "Fe": 0.1,
            "Si": 0.4, "Se": 0.2,
            "Ni": 0.6, "Ne": 0.5
        }

    def _load_mapping(self) -> dict:
        """加载 MBTI 映射文件，若文件不存在则返回空字典"""
        if os.path.exists(self.mapping_file):
            try:
                with open(self.mapping_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载 MBTI 映射文件失败: {e}")
        return {}

    def _load(self) -> dict:
        if os.path.exists(self.profile_file):
            try:
                with open(self.profile_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save(self):
        with open(self.profile_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def get_preferred_function(self, task_type: str = None) -> Tuple[str, float]:

        """
            返回当前置信度最高的认知功能及其置信度。动态调整最高功能
            忽略 task_type 参数，直接使用 cognitive_functions 中的最大值。
        """
        functions = self.data["cognitive_functions"]
        # 找到置信度最高的功能
        func = max(functions, key=functions.get)
        return func, functions[func]

        # """
        # 根据任务类型和用户偏好，返回最适合的认知功能缩写及其置信度分数。
        # """
        # functions = self.data["cognitive_functions"]
        # if task_type == "knowledge":
        #     ti = functions.get("Ti", 0)
        #     te = functions.get("Te", 0)
        #     func = "Ti" if ti >= te else "Te"
        #     return func, functions[func]
        # elif task_type == "emotional":
        #     fi = functions.get("Fi", 0)
        #     fe = functions.get("Fe", 0)
        #     func = "Fi" if fi >= fe else "Fe"
        #     return func, functions[func]
        # else:
        #     func = max(functions, key=functions.get)
        #     return func, functions[func]

    def update_function_confidence(self, function: str, delta: float, context: str = "",
                                   question: str = "", answer: str = "", old_confidence: float = None):
        """增加或减少某个功能的置信度，并记录反馈日志"""
        if function not in self.data["cognitive_functions"]:
            return
        old = self.data["cognitive_functions"][function]
        new = max(0.0, min(1.0, old + delta))
        self.data["cognitive_functions"][function] = new
        self.data["last_updated"] = time.time()
        self._save()
        # 记录详细日志
        self._log_feedback(
            function=function,
            feedback="like" if delta > 0 else "dislike",
            delta=delta,
            context=context,
            question=question[:100] if question else "",
            answer=answer[:100] if answer else "",
            old_confidence=old,
            new_confidence=new
        )

    def decay_confidences(self, decay_rate: float = 0.01):
        """对所有功能置信度进行衰减，长时间未用逐渐降低"""
        now = time.time()
        last = self.data.get("last_updated", now)
        hours_passed = (now - last) / 3600
        if hours_passed < 1:  # 小于1小时不衰减
            return
        decay = decay_rate * (hours_passed / 24)  # 按天衰减
        for func in self.data["cognitive_functions"]:
            old = self.data["cognitive_functions"][func]
            new = max(0.0, old - decay)
            self.data["cognitive_functions"][func] = new
        self.data["last_updated"] = now
        self._save()

    def _log_feedback(self, function: str, feedback: str, delta: float, context: str,
                      question: str, answer: str, old_confidence: float, new_confidence: float):
        entry = {
            "timestamp": time.time(),
            "function": function,
            "feedback": feedback,
            "delta": delta,
            "context": context,
            "question": question,
            "answer": answer,
            "old_confidence": old_confidence,
            "new_confidence": new_confidence
        }
        logs = self._load_feedback_log()
        logs.append(entry)
        with open(self.feedback_log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

    def _load_feedback_log(self) -> List[Dict]:
        if os.path.exists(self.feedback_log_file):
            try:
                with open(self.feedback_log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return []

    def get_recent_feedback_stats(self, recent_n: int = 20) -> Dict[str, Dict]:
        logs = self._load_feedback_log()
        recent = logs[-recent_n:] if len(logs) > recent_n else logs
        stats = {}
        for entry in recent:
            func = entry["function"]
            delta = entry["delta"]
            if func not in stats:
                stats[func] = {"positive": 0, "negative": 0}
            if delta > 0:
                stats[func]["positive"] += 1
            elif delta < 0:
                stats[func]["negative"] += 1
        return stats

    def recalibrate_from_feedback(self, recent_n: int = 20, base_confidence: float = 0.5):
        stats = self.get_recent_feedback_stats(recent_n)
        total_feedbacks = sum(v["positive"] + v["negative"] for v in stats.values())
        if total_feedbacks == 0:
            return
        for func, counts in stats.items():
            net = counts["positive"] - counts["negative"]
            adjustment = (net / total_feedbacks) * 0.5
            new_val = max(0.0, min(1.0, base_confidence + adjustment))
            self.data["cognitive_functions"][func] = new_val
        self.data["last_updated"] = time.time()
        self._save()

    @staticmethod
    def init_from_mbti(mbti_type: str, mapping_file: str = "mbti_mapping.json") -> dict:
        """
        根据 MBTI 类型返回初始的 cognitive_functions 字典。
        优先从 mapping_file 加载，否则使用内置映射。
        """
        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                return mapping.get(mbti_type.upper(), mapping.get("INTP", {}))
            except Exception:
                pass
        # 内置简单映射（兼容旧代码）
        default_map = {
            "INTP": {"Ti": 0.8, "Te": 0.2, "Fi": 0.3, "Fe": 0.1, "Si": 0.4, "Se": 0.2, "Ni": 0.6, "Ne": 0.5},
            "INTJ": {"Ni": 0.8, "Ne": 0.3, "Te": 0.7, "Ti": 0.2, "Fi": 0.3, "Fe": 0.1, "Se": 0.4, "Si": 0.2},
            # 可根据需要扩展其他类型
        }
        return default_map.get(mbti_type.upper(), default_map["INTP"])