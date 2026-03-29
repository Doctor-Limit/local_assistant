# 透明智能体：一个可验证、可调节、可解释的AI助手

> **“我能不能亲手验证AI在想什么？——现在，你可以了。”**

这是一个从赌约开始的荒诞项目。为了回应一个AI的挑衅，我赌气搭建了这个智能体，却发现我们真正需要的，是一个每一步都敢说“为什么”的透明助手。

它会把记忆像档案一样给你翻看，会让AI的性格变成一堆可拧的旋钮，会把解释分成三个层次——从“一句话结论”到“完整推理链”再到“连底裤都给你看的JSON”。它可能不够“像人”，但它敢说：“我为什么这么想。”

**在线演示**：[http://43.132.194.13:8501/](http://43.132.194.13:8501/)（节点在香港，如果没崩的话）  
**GitHub仓库**：[https://github.com/Doctor-Limit/local_assistant](https://github.com/Doctor-Limit/local_assistant)

---

## 🧠 核心功能

- **可追溯的记忆**  
  每条记忆都有唯一ID，侧边栏列出所有记忆，点击即可查看完整内容。AI引用记忆时必须带上ID，是真是假，一看便知。

- **可调节的画像**  
  基于荣格八维认知功能（Ti, Te, Fi, Fe, Si, Se, Ni, Ne），每个功能配有0~1的滑块。你拖一拖，AI的回答风格立马改变——从数学老师般的逻辑推导，到温柔的情感共情，随你定义。

- **分层次的解释**  
  - **简洁型**：一句话结论，适合只想快速知道结果的人。  
  - **逻辑型**：完整推理步骤 + Mermaid流程图，适合想理解“为什么”的人。  
  - **调试型**：完整JSON，包含置信度、引用来源、工具调用记录，适合开发者。

- **RAG检索增强**  
  支持向量检索、混合检索（向量+TF‑IDF）、重排序（Cross‑Encoder）。技术类问题会自动检索知识库，给出有据可依的回答。

- **工具调用**  
  内置天气、时间、计算、安全命令等工具，支持多轮调用（例如“先查北京天气，再计算25*4”）。

- **长期记忆与反馈学习**  
  高置信度记忆自动存入长期记忆（ChromaDB），支持语义检索。点赞/点踩会实时调整认知功能置信度，画像会逐渐贴近你的偏好。

- **多模式体验**  
  侧边栏提供MBTI初始测试、手动调整认知功能、查看反馈记录、检索长期记忆、查看记忆详情等，交互友好。

---

## 🚀 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/Doctor-Limit/local_assistant.git
cd local_assistant
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```
*注：`requirements.txt` 尚未提供，请根据 `import` 手动安装主要包：*
- `streamlit`
- `chromadb`
- `sentence-transformers`
- `cachetools`
- `scikit-learn`（可选，用于混合检索）
- `requests`
- `python-dotenv`
- `simpleeval`

### 3. 配置API（默认使用云端硅基流动API）
项目根目录下新建 `.env`，填入你的 `ASSISTANT_CLOUD_API_KEY`。  
若想使用本地Ollama，修改 `config.py` 中 `use_cloud_api = False`，并确保Ollama已启动，模型为 `qwen2.5:1.5b`。

### 4. 运行
```bash
streamlit run app_streamlit.py
```

浏览器会自动打开 `http://localhost:8501`。首次加载需要下载嵌入模型（约30秒），之后即可开始对话。

---

## 📖 使用指南

### 初始化画像
首次打开会要求完成简易MBTI测试，只需选择几个描述，系统会初始化认知功能置信度。你也可以直接“跳过”，使用默认INTP画像。

### 调整AI风格
侧边栏 → “✏️ 手动调整认知功能”，拖动滑块即可实时改变各功能权重。AI会优先选择置信度最高的功能来回答。

### 选择解释模式
侧边栏 → “解释模式”，可选简洁型、逻辑型、调试型。逻辑型会显示Mermaid流程图，调试型可下载JSON。

### 查看记忆
侧边栏 → “记忆库” → “短期记忆”，点击“查看”可弹窗显示完整内容。在对话中，AI引用记忆时会带上ID，你可以直接点击验证。

### 反馈与校准
每轮对话下方有“有帮助/没帮助”按钮，点击后会调整对应认知功能的置信度。侧边栏的“反馈与校准”区域可查看历史反馈，并一键基于近期反馈重新校准画像。

### 示例问题
界面上方有四个示例按钮：  
- “石头剪刀布策略”  
- “HashMap 线程安全”  
- “电车难题”  
- “工作选择”  

点击即可快速体验不同风格的回答。

---

## 🗂 项目结构

```
local_assistant/
├── app_streamlit.py            # Streamlit 主界面
├── assistant_pkg/              # 核心模块
│   ├── assistant.py            # 助手主控制器
│   ├── user_profile.py         # 用户画像（MBTI + 认知功能）
│   ├── memory.py               # 短期/长期记忆管理
│   ├── retriever.py            # RAG检索器（向量、混合、重排序）
│   ├── llm.py                  # LLM引擎（云端/Ollama）
│   ├── explainer.py            # 解释生成器（自然语言 + 推理链）
│   ├── explanation_router.py   # 解释路由（风格选择）
│   ├── answer_explanation_generator.py  # 答案+解释合并生成
│   ├── response.py             # 工具调用响应处理
│   ├── tools.py                # 注册的工具函数
│   ├── config.py               # 配置管理
│   ├── rules.py                # 规则引擎
│   ├── state.py                # 状态管理
│   └── cache.py                # 解释缓存
├── chroma_db/                  # ChromaDB持久化目录
├── test_rag.txt                # 示例知识库
├── mbti_mapping.json           # MBTI→认知功能映射
├── few_shot_examples.json      # Few-shot示例（工具调用）
├── style_examples.json         # 风格示例（场景匹配）
└── .env                        # 环境变量（API Key）
```

---

## ⚙️ 技术栈

- **前端**：Streamlit
- **后端**：Python
- **向量数据库**：ChromaDB
- **嵌入模型**：sentence-transformers/all-MiniLM-L6-v2
- **重排序**：cross-encoder/ms-marco-MiniLM-L-6-v2
- **LLM**：硅基流动 API（Qwen2.5-7B） 或 本地 Ollama（Qwen2.5:1.5B）
- **记忆检索**：TF‑IDF（可选）+ 余弦相似度

---

## 🔮 当前限制与未来计划

### 已知问题
- 差异化风格体现不够明显，有时不同认知功能回答相似。
- 复杂语句工具调用不稳定，多步推理易出错。
- 仅支持单用户会话，无并发。
- 记忆提炼较粗糙，跨会话知识保留不够准确。
- 机械解释（权重可视化）暂不支持。
- 长时间对话可能存在迎合倾向。
- 用户反馈尚未用于模型训练，仅用于置信度调整。

### 下一步计划
- 引入元认知自省，让AI反思“我是不是在迎合”。
- 模块化重构，参考CIRISAgent等架构。
- 增加消息队列支持多实例并发。
- 优化记忆提炼算法。
- 探索特征归因可视化（如SHAP）。

---

## 🤝 贡献与反馈

欢迎提交Issue报告问题或建议。暂不接受PR（代码还能跑，怕一改就崩），但如果你有好的想法，欢迎在Issue中讨论。

**GitHub Issue**：[https://github.com/Doctor-Limit/local_assistant/issues](https://github.com/Doctor-Limit/local_assistant/issues)

---

## 📄 许可

本项目遵循 MIT 协议（请根据实际情况添加许可证文件）。

---

## 🙏 致谢

- 荣格八维理论，以及所有在可解释AI领域探索的研究者。
- 硅基流动提供的云端API服务。
- 开源社区的所有工具与库。

---

**最后，回到最初那个问题：“我能不能亲手验证AI在想什么？”**

——现在，你可以了。
