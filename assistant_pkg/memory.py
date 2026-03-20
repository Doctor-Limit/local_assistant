# memory.py
import time
import json
from collections import deque
from typing import List, Tuple

class MemoryItem:
    """单条记忆"""
    def __init__(self, role: str, content: str, timestamp: float = None):
        self.role = role          # 'user', 'assistant', 'system', 'tool'
        self.content = content
        self.timestamp = timestamp or time.time()

    def to_dict(self):
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}

    @classmethod
    def from_dict(cls, d):
        return cls(d["role"], d["content"], d["timestamp"])


class MemoryManager:
    """记忆管理器，维护短期记忆（对话历史）和可选的长期记忆（此处简化）"""
    def __init__(self, max_size: int = 50, memory_file: str = None):
        self.max_size = max_size
        self.memory_file = memory_file
        self.short_term = deque(maxlen=max_size)  # 存储 MemoryItem 对象
        if memory_file:
            self.load()

    def add(self, item: MemoryItem):
        self.short_term.append(item)
        if self.memory_file:
            self.save()

    def get_recent(self, n: int = None) -> List[MemoryItem]:
        """获取最近 n 条记忆，默认全部（按时间顺序）"""
        if n is None:
            return list(self.short_term)
        return list(self.short_term)[-n:]

    def clear(self):
        """清空记忆"""
        self.short_term.clear()
        if self.memory_file:
            self.save()

    def save(self):
        """保存到文件"""
        if not self.memory_file:
            return
        data = [item.to_dict() for item in self.short_term]
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存记忆失败: {e}")

    def load(self):
        """从文件加载"""
        if not self.memory_file:
            return
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.short_term.clear()
                for d in data:
                    self.short_term.append(MemoryItem.from_dict(d))
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"加载记忆失败: {e}")

