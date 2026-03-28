# cache.py
import hashlib
import json
from functools import lru_cache
from typing import Optional
from cachetools import TTLCache

class ExplanationCache:
    def __init__(self, maxsize=1000, ttl=3600):
        """使用 TTL 缓存，默认最大条目 1000，过期时间 1 小时"""
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)  # 简单内存缓存，可用 Redis 替代

    def get_key(self, question: str, user_profile: dict, function: str) -> str:
        """生成缓存键：问题 + 用户画像（认知功能强度）"""
        # 归一化问题
        norm_question = question.strip().lower()
        # 画像取 cognitive_functions 的排序值，确保一致性
        funcs = user_profile.get("cognitive_functions", {})
        # 直接使用原始值，不进行舍入，浮点数精度可能导致过长字符串
        # 将其转为 JSON 字符串，保留足够精度
        func_str = json.dumps({k: v for k, v in sorted(funcs.items())}, sort_keys=True)
        return hashlib.md5((norm_question + func_str + function).encode()).hexdigest()

    def get(self, key: str) -> Optional[dict]:
        return self.cache.get(key)

    def set(self, key: str, value: dict):
        self.cache[key] = value

# 全局缓存实例
cache = ExplanationCache()