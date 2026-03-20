# state.py
import json
import time

class State:
    """状态管理器，保存会话状态（如情绪、轮数等）"""
    def __init__(self, state_file: str = None):
        self.state_file = state_file
        self.data = {
            "initialized": False,
            "turn_count": 0,
            "mood": "neutral",
            "last_input_time": None,
        }
        if state_file:
            self.load()

    def set(self, key, value):
        self.data[key] = value
        if self.state_file:
            self.save()

    def get(self, key, default=None):
        return self.data.get(key, default)

    def increment(self, key, amount=1):
        self.data[key] = self.data.get(key, 0) + amount
        if self.state_file:
            self.save()

    def save(self):
        if not self.state_file:
            return
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存状态失败: {e}")

    def load(self):
        if not self.state_file:
            return
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                self.data.update(json.load(f))
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"加载状态失败: {e}")

