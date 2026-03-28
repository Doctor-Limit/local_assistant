# memory.py
import time
import json
import chromadb
import uuid
import numpy as np
from chromadb.utils import embedding_functions
from collections import deque
from typing import List, Dict, Optional

class MemoryItem:
    """单条记忆"""

    def __init__(self, role: str, content: str, timestamp: float = None,
                 source_type: str = "conversation",  # conversation, rag, tool, user_feedback, refined
                 source_id: str = None,  # 关联ID，如对话轮次、知识片段ID、工具名
                 confidence: float = 1.0,  # 0~1
                 metadata: dict = None,  # 额外信息，如检索得分、工具参数
                 mem_id: Optional[str] = None,
                 embedding: Optional[List[float]] = None):
        self.role = role
        self.content = content
        self.timestamp = timestamp or time.time()
        self.source_type = source_type
        self.source_id = source_id
        self.confidence = confidence
        self.metadata = metadata or {}
        self.embedding = embedding
        self.mem_id = mem_id or f"mem_{int(self.timestamp * 1000)}_{uuid.uuid4().hex[:8]}"

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "mem_id": self.mem_id
        }


    @classmethod
    def from_dict(cls, d):
        return cls(
            role=d["role"],
            content=d["content"],
            timestamp=d["timestamp"],
            source_type=d.get("source_type", "conversation"),
            source_id=d.get("source_id"),
            confidence=d.get("confidence", 1.0),
            metadata=d.get("metadata", {}),
            mem_id=d.get("mem_id")
        )


class LongTermMemory:
    """长期记忆：存储从短期记忆中提炼出的重要信息，支持语义检索"""
    def __init__(self, collection_name="long_term_memory", persist_directory="./chroma_db",
                 model_name="all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def add(self, memory_item: MemoryItem):
        """添加一条长期记忆"""
        doc_id = memory_item.mem_id  # 使用记忆的唯一 ID
        self.collection.add(
            documents=[memory_item.content],
            metadatas=[{
                "mem_id": memory_item.mem_id,
                "source_type": memory_item.source_type,
                "source_id": memory_item.source_id,
                "confidence": memory_item.confidence,
                "role": memory_item.role,
                "timestamp": memory_item.timestamp
            }],
            ids=[doc_id]
        )

    def search(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        results = self.collection.query(query_texts=[query], n_results=top_k)
        items = []
        for doc, meta, id_ in zip(results['documents'][0], results['metadatas'][0], results['ids'][0]):
            item = MemoryItem(
                role=meta.get("role", "system"),
                content=doc,
                timestamp=float(meta.get("timestamp", 0)),
                source_type=meta.get("source_type", "long_term"),
                source_id=meta.get("source_id", id_),
                confidence=float(meta.get("confidence", 0.5)),
                metadata=meta,
                mem_id=meta.get("mem_id", id_)  # 使用保存的 mem_id
            )
            items.append(item)
        return items

class MemoryManager:
    """记忆管理器，维护短期记忆（对话历史）和可选的长期记忆（此处简化）"""
    def __init__(self, max_size: int = 50, memory_file: str = None,
                 long_term_enabled: bool = True,
                 long_term_persist_dir: str = "./chroma_db",
                 embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.max_size = max_size
        self.memory_file = memory_file
        self.short_term = deque(maxlen=max_size)
        self.long_term = LongTermMemory(persist_directory=long_term_persist_dir) if long_term_enabled else None
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model_name)
        if memory_file:
            self.load()

    def update_confidence(self, mem_id: str, new_confidence: float):
        for item in self.short_term:
            if item.mem_id == mem_id:
                item.confidence = new_confidence
                self.save()
                return True
        if self.long_term:
            # 更新长期记忆（ChromaDB）
            self.long_term.collection.update(
                ids=[mem_id],
                metadatas=[{"confidence": new_confidence}]
            )
            return True
        return False

    def get_by_id(self, mem_id: str) -> Optional[MemoryItem]:
        # 先查短期记忆
        for item in self.short_term:
            if getattr(item, 'mem_id', None) == mem_id:
                return item
        # 再查长期记忆
        if self.long_term:
            # 长期记忆目前不支持按 ID 快速检索，需遍历或利用 ChromaDB 的 get 方法
            # 目前采用 get(ids=[mem_id])
            result = self.long_term.collection.get(ids=[mem_id])
            if result['documents']:
                doc = result['documents'][0]
                meta = result['metadatas'][0]
                return MemoryItem(
                    role=meta.get("role", "system"),
                    content=doc,
                    timestamp=float(meta.get("timestamp", 0)),
                    source_type=meta.get("source_type", "long_term"),
                    source_id=meta.get("source_id", mem_id),
                    confidence=float(meta.get("confidence", 0.5)),
                    metadata=meta,
                    mem_id=mem_id
                )
        return None

    def add(self, item: MemoryItem):
        # 生成嵌入（如果未提供）
        if item.embedding is None and hasattr(self, 'embedding_fn'):
            item.embedding = self.embedding_fn([item.content])[0].tolist()
        self.short_term.append(item)
        # 如果是重要记忆（高置信度且非对话），自动存入长期记忆
        if self.long_term and item.confidence > 0.7 and item.source_type != "conversation":
            self.long_term.add(item)
        if self.memory_file:
            self.save()

    # 在 assistant.py 中，当生成解释后，将推理链存入长期记忆
    def store_reasoning_chain(self, explanation_data: Dict, context: Dict):
        # 提取关键信息
        chain_text = explanation_data.get("text", "")
        mem_item = MemoryItem(
            role="assistant",
            content=chain_text,
            source_type="reasoning_chain",
            confidence=0.8,
            metadata={
                "question": context.get("user_input", ""),
                "answer": context.get("final_answer", ""),
                "function": explanation_data.get("function")  # 如果有
            }
        )
        self.memory.add(mem_item)  # 这会自动存入短期记忆，且高置信度会转入长期记忆

    def search_short_term(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        """语义检索短期记忆，返回最相似的条目"""
        if not self.short_term:
            return []
        # 如果有嵌入模型且短期记忆中有嵌入，使用向量检索
        if hasattr(self, 'embedding_fn') and any(item.embedding for item in self.short_term):
            query_embedding = self.embedding_fn([query])[0]
            # 计算余弦相似度
            scores = []
            for item in self.short_term:
                if item.embedding is not None:
                    # 计算余弦相似度
                    sim = np.dot(query_embedding, item.embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(item.embedding))
                    scores.append((sim, item))
                else:
                    scores.append((0, item))  # 没有嵌入的降级
            scores.sort(key=lambda x: x[0], reverse=True)
            return [item for _, item in scores[:top_k]]
        else:
            # 降级到关键词匹配（原有逻辑）
            query_words = set(query.lower().split())
            scored = []
            for item in self.short_term:
                item_words = set(item.content.lower().split())
                overlap = len(query_words & item_words)
                if overlap > 0:
                    score = overlap / max(len(query_words), len(item_words))
                    scored.append((score, item))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [item for _, item in scored[:top_k]]

    def search(self, query: str, top_k: int = 5, include_long_term: bool = True) -> List[MemoryItem]:
        """综合检索短期和长期记忆"""
        short_results = self.search_short_term(query, top_k)
        if include_long_term and self.long_term:
            long_results = self.long_term.search(query, top_k)
            # 合并去重（按内容去重）
            seen = set()
            all_results = []
            for item in short_results + long_results:
                if item.content not in seen:
                    seen.add(item.content)
                    all_results.append(item)
            return all_results[:top_k]
        return short_results[:top_k]

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

