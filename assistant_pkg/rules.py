# rules.py
import re

class RuleEngine:
    """简单的规则引擎，通过正则匹配预设回复"""
    def __init__(self):
        self.rules = [
            (r"你好|您好|在吗", "你好！我是你的本地助手，有什么可以帮你的？"),
            (r"再见|拜拜| bye", "再见！期待下次为你服务。"),
            (r"谢谢|感谢", "不客气，很高兴能帮到你。"),
            (r"你是谁|你叫什么", "我是你的本地助手小莲，一个温柔细心的AI助手。"),
        ]

    def check(self, text: str) -> str or None:
        """检查是否匹配规则，返回回复或None"""
        for pattern, response in self.rules:
            if re.search(pattern, text, re.IGNORECASE):
                return response
        return None

