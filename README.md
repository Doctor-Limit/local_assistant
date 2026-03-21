# 🧠 Local LLM Assistant

> **在线体验**：[http://43.132.194.13:8501](http://43.132.194.13:8501)  
> **GitHub**：[https://github.com/Doctor-Limit/local_assistant](https://github.com/Doctor-Limit/local_assistant)

一个基于本地大模型的模块化智能体框架。支持多轮对话、检索增强生成（RAG）和工具调用，全部运行在本地（通过 Ollama + Qwen2.5），无需依赖外部 API。  
项目已部署至云服务器，可在线体验。

**项目状态**：积极开发中，核心功能已可用，正在持续优化性能和稳定性。

---

## ✨ 当前功能

- ✅ 本地模型部署（Ollama + Qwen2.5 1.5B 量化版）
- ✅ 模块化架构：记忆管理、规则引擎、状态控制、LLM 引擎等
- ✅ 多轮对话（上下文记忆）
- ✅ 检索增强生成（RAG）：基于本地知识库回答问题，响应时间已优化至 3 秒内
- ✅ 流式输出（逐字返回，提升体验）
- ✅ 基础工具调用（天气、时间、计算、安全命令）
- ✅ 通过 Streamlit 提供 Web 界面
- ✅ 支持云端 API 一键切换（如硅基流动）

---

## 🛠️ 技术栈

- **模型部署**：Ollama + Qwen2.5:1.5b
- **后端**：Python 3.10+，Streamlit
- **检索**：ChromaDB 向量数据库 + sentence-transformers（all-MiniLM-L6-v2）
- **容器化**：Docker（可选）

---

## 📦 项目结构

local_assistant/
├── assistant_pkg/ # 核心模块
│ ├── assistant.py # 主控制器
│ ├── config.py # 配置管理
│ ├── llm.py # LLM 引擎（Ollama/云端）
│ ├── memory.py # 记忆管理
│ ├── retriever.py # RAG 检索（ChromaDB）
│ ├── rules.py # 规则引擎
│ ├── state.py # 状态管理
│ ├── tools.py # 工具函数
│ └── response.py # 响应生成器
├── app_streamlit.py # Streamlit Web 界面
├── main.py # 命令行交互
├── test_rag.txt # 示例知识库（Java 八股文）
├── requirements.txt
└── README.md


---

## ⚙️ 配置

可通过 `config.json` 或环境变量（前缀 `ASSISTANT_`）修改配置。主要配置项如下：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `model` | Ollama 模型名称 | `qwen2.5:1.5b` |
| `temperature` | 生成温度 | `0.7` |
| `rag_enable` | 是否启用 RAG | `True` |
| `knowledge_file` | 知识库文件路径 | `test_rag.txt` |
| `rag_top_k` | 检索返回的文档数 | `3` |
| `memory_size` | 记忆窗口大小 | `50` |
| `use_cloud_api` | 是否使用云端 API | `False` |
| `cloud_api_key` | 云端 API 密钥 | （需自行填写） |
| `cloud_model` | 云端模型名称 | `Qwen/Qwen2.5-7B-Instruct` |

详细配置见 `assistant_pkg/config.py`。

---

## 🚀 快速开始

### 环境要求
- Python 3.10+
- Ollama 已安装并运行（[ollama.com](https://ollama.com)）
- 拉取模型：`ollama pull qwen2.5:1.5b-instruct-q4_0`

### 安装与运行
```bash
git clone https://github.com/Doctor-Limit/local_assistant.git
cd local_assistant
pip install -r requirements.txt
streamlit run app_streamlit.py

访问终端显示的本地地址（如 http://localhost:8501）即可开始对话。


💡 使用示例
Web 界面

    输入问题，助手会流式返回答案。

    若启用 RAG，会自动检索知识库相关内容（示例知识库为 Java 八股文）。

工具调用示例

    时间：现在几点？

    天气：北京天气

    计算：计算 25*4+3

    安全命令：执行 ls（仅限白名单命令）

❓ 常见问题

首次运行很慢？

Ollama 首次加载模型需要几秒，后续对话会很快。可设置模型常驻内存：
```bash
ollama serve --keep-alive 600

知识库如何更新？
将文本文件（按空行或 Q： 格式分割）放入 knowledge_file 指定路径，程序启动时自动加载。如需动态添加，可调用 Retriever.add_documents() 方法。

如何切换云端模型？
在 config.py 中设置 use_cloud_api = True，并配置 cloud_api_key 和 cloud_base_url（OpenAI 兼容格式）。目前支持硅基流动等平台。

偶尔响应失败或超时？
可能是本地模型加载延迟或资源不足。建议增加 timeout 设置，或保持模型常驻内存。若问题持续，请提交 Issue。

## 🚧 下一步计划

- **可解释AI**：尝试实现工具调用解释、RAG检索解释、最终答案置信度/依据、对话链可视化。
- **复杂语句工具调用**：当前在用户意图模糊或多步推理时，模型可能无法正确触发工具；计划通过 **few-shot 示例** 和 **function calling 强化** 来优化。
- **RAG 检索精度**：目前使用 ChromaDB + 句子向量，偶有检索不相关结果；下一步引入 **query 改写**、**重排序（reranking）** 提升准确性。
- **多实例并发**：当前为单线程，暂不支持高并发；后续将引入异步处理与多进程支持。
- **记忆持久化**：短期记忆已实现，长期记忆计划采用 **向量数据库 + 摘要** 方式存储。

---

🤝 贡献
欢迎提交 Issue 和 Pull Request。开发前请确保代码通过基础测试，并遵循现有代码风格。
