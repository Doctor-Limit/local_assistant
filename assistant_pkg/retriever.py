# retriever.py
import os
import re
import chromadb
import numpy as np
from chromadb.utils import embedding_functions
from typing import List, Tuple, Optional

# 可选依赖
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    RERANK_AVAILABLE = True
except ImportError:
    RERANK_AVAILABLE = False


class Retriever:
    def __init__(self, collection_name="java_knowledge", persist_directory="./chroma_db",
                 model_name="all-MiniLM-L6-v2",
                 use_hybrid: bool = True,        # 是否启用混合检索
                 use_rerank: bool = True,        # 是否启用重排序
                 rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        增强版检索器，支持向量检索、混合检索（向量+关键词）、重排序。
        :param collection_name: 集合名称
        :param persist_directory: ChromaDB 持久化目录
        :param model_name: 嵌入模型名称
        :param use_hybrid: 是否启用混合检索（向量+TF-IDF）
        :param use_rerank: 是否启用重排序（需要 sentence-transformers）
        :param rerank_model: 重排序模型名称
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )
        self.use_hybrid = use_hybrid
        self.use_rerank = use_rerank

        # 混合检索相关
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.documents_for_bm25 = []  # 存储所有文档，用于关键词检索

        # 重排序相关，改用cpu重排序
        self.reranker = None
        if use_rerank and RERANK_AVAILABLE:
            try:
                # 直接在这里指定 device='cpu'
                self.reranker = CrossEncoder(rerank_model, device='cpu')
            except Exception as e:
                print(f"加载重排序模型失败: {e}，将禁用重排序")
                self.use_rerank = False
        elif use_rerank and not RERANK_AVAILABLE:
            print("未安装 sentence-transformers，重排序功能不可用。请执行 'pip install sentence-transformers'")
            self.use_rerank = False

        # 提示依赖缺失
        if use_hybrid and not SKLEARN_AVAILABLE:
            print("未安装 scikit-learn，混合检索中的关键词部分将不可用。请执行 'pip install scikit-learn'")
            self.use_hybrid = False

    def add_documents(self, documents: List[str], metadatas: List[dict] = None, ids: List[str] = None):
        """添加文档到向量库，并更新混合检索索引"""
        if ids is None:
            ids = [str(i) for i in range(len(documents))]
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

        # 更新混合检索索引
        if self.use_hybrid and SKLEARN_AVAILABLE:
            self.documents_for_bm25 = documents
            if self.tfidf_vectorizer is None:
                self.tfidf_vectorizer = TfidfVectorizer(stop_words=None, token_pattern=r"(?u)\b\w+\b")
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)


    def _search_vector(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """向量检索，返回 (文档, 相似度) 列表"""
        results = self.collection.query(query_texts=[query], n_results=top_k)
        if not results['documents']:
            return []
        docs = results['documents'][0]
        # Chroma 返回的 distances 越小越相似，转换为相似度
        distances = results['distances'][0] if 'distances' in results else [0.5] * len(docs)
        similarities = [1 - d for d in distances]
        return list(zip(docs, similarities))

    def _search_keyword(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """关键词检索（TF-IDF 余弦相似度）"""
        if not self.documents_for_bm25 or self.tfidf_matrix is None:
            return []
        query_vec = self.tfidf_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(self.documents_for_bm25[i], scores[i]) for i in top_indices if scores[i] > 0]

    def _hybrid_search(self, query: str, top_k: int = 3) -> List[str]:
        """混合检索：向量 + 关键词，加权合并"""
        vec_results = self._search_vector(query, top_k=top_k)
        kw_results = self._search_keyword(query, top_k=top_k)

        doc_scores = {}
        for doc, score in vec_results:
            doc_scores[doc] = doc_scores.get(doc, 0) + score * 0.6   # 向量权重 0.6
        for doc, score in kw_results:
            doc_scores[doc] = doc_scores.get(doc, 0) + score * 0.4   # 关键词权重 0.4

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs[:top_k]]

    def _rerank(self, query: str, documents: List[str], top_k: int = 3) -> List[str]:
        """使用交叉编码器重排序"""
        if not self.reranker or not documents:
            return documents[:top_k]
        pairs = [(query, doc) for doc in documents]
        scores = self.reranker.predict(pairs)
        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:top_k]]

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """
        统一检索接口：
        1. 根据配置选择纯向量或混合检索
        2. 如果启用重排序，对候选结果进行重排序
        """
        if self.use_hybrid and SKLEARN_AVAILABLE:
            # 混合检索，先取更多候选（top_k*2）供重排序使用
            candidate_k = top_k * 2 if self.use_rerank else top_k
            candidates = self._hybrid_search(query, top_k=candidate_k)
        else:
            # 纯向量检索
            vec_results = self._search_vector(query, top_k=top_k)
            candidates = [doc for doc, _ in vec_results]

        if self.use_rerank and self.reranker:
            return self._rerank(query, candidates, top_k=top_k)
        else:
            return candidates[:top_k]

    def load_from_file(self, filepath: str):
        """从文件加载知识库，支持多种分块方式"""
        if not os.path.exists(filepath):
            print(f"知识库文件不存在: {filepath}")
            return
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # 分块策略（优先级递减）
        if 'Q：' in content or 'Q:' in content:
            # 按 "Q：" 或 "Q:" 分割（保留问题标识）
            chunks = re.split(r'\n(?=Q[：:])', content)
        else:
            # 按空行分割
            chunks = [c.strip() for c in content.split('\n\n') if c.strip()]
        if not chunks:
            # 固定长度分块（带重叠）
            chunk_size = 500
            overlap = 50
            chunks = []
            for i in range(0, len(content), chunk_size - overlap):
                chunks.append(content[i:i+chunk_size])
        chunks = [c.strip() for c in chunks if c.strip()]
        self.add_documents(chunks)
        print(f"已加载 {len(chunks)} 个知识块")