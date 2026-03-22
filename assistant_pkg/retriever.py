# retriever.py
import os
import re
import chromadb
from chromadb.utils import embedding_functions
from typing import List

class Retriever:
    def __init__(self, collection_name="java_knowledge", persist_directory="./chroma_db",
                 model_name="all-MiniLM-L6-v2"):
        #paraphrase-multilingual-MiniLM-L12-v2 这个中文适配更好但是响应较慢
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def add_documents(self, documents: List[str], metadatas: List[dict] = None, ids: List[str] = None):
        """添加文档到集合"""
        if ids is None:
            ids = [str(i) for i in range(len(documents))]
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """检索最相关的文档内容"""
        results = self.collection.query(query_texts=[query], n_results=top_k)
        if results['documents']:
            return results['documents'][0]
        return []

    def load_from_file(self, filepath: str):
        """从文本文件加载知识库（按空行分割成段落）"""
        if not os.path.exists(filepath):
            print(f"知识库文件不存在: {filepath}")
            return
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        # 按 Q： 作为分割标记，使用正则表达式
        chunks = re.split(r'\n(?=Q[：:])', content)
        chunks = [c.strip() for c in chunks if c.strip()]

        self.add_documents(chunks)

