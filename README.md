# 🧠 Local LLM Assistant

一个基于本地大模型的模块化智能体框架。支持多轮对话、检索增强生成（RAG）和工具调用，全部运行在本地（通过 Ollama + Qwen2.5），无需依赖外部 API。

**项目状态**：积极开发中，核心功能已可用，正在持续优化性能和稳定性。

---

## ✨ 当前功能

- ✅ 本地模型部署（Ollama + Qwen2.5 1.5B 量化版）
- ✅ 模块化架构：记忆管理、规则引擎、状态控制、LLM 引擎等
- ✅ 多轮对话（上下文记忆）
- ✅ 检索增强生成（RAG）：基于本地知识库回答问题，响应时间已优化至 3 秒内
- ✅ 流式输出（逐字返回，提升体验）
- ✅ 基础工具调用（天气、时间等）
- ✅ 通过 Streamlit 提供 Web 界面

---

## 🛠️ 技术栈

- **模型部署**：Ollama + qwen2.5:1.5b-instruct-q4_0
- **后端**：Python 3.10+，Streamlit
- **检索**：关键词匹配 
- **容器化**：Docker（可选）

---

### 环境要求
- Python 3.10+
- Ollama 已安装并运行（[ollama.com](https://ollama.com)）
- 拉取模型：`ollama pull qwen2.5:1.5b-instruct-q4_0`

### 安装与运行
```bash
git clone https://github.com/Doctor-Limit/local_assistant.git
cd local_assistant
pip install -r requirements.txt
streamlit run app.py
