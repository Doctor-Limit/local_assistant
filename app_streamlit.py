import streamlit as st
from assistant_pkg.assistant import Assistant

# 初始化助手（全局单例）
@st.cache_resource
def get_assistant():
    return Assistant()

assistant = get_assistant()

# 设置页面标题
st.set_page_config(page_title="本地智能体助手", page_icon="🤖")
st.title("🤖 本地智能体助手")
st.info(
    "🚀 **演示说明**：\n\n"
    "本智能体当前为提升响应速度与回答质量，使用 **云端 API**（硅基流动 Qwen2.5-7B）进行推理。\n"
    "项目代码完整支持 **本地模型切换**（Ollama + Qwen2.5:1.5B），并可一键回退。\n\n"
    "⚙️ **技术亮点**：模块化架构 · 工具调用 · RAG 检索 · 流式输出 · 对话记忆"
)

# 可选：折叠的待优化项
with st.expander("📝 已知优化项 & 下一步计划"):
    st.markdown("""
    - **可解释AI**：尝试实现以下功能：工具调用解释，RAG检索解释，最终答案的置信度/依据，对话链可视化。
    - **复杂语句工具调用**：当前在用户意图模糊或多步推理时，模型可能无法正确触发工具；计划通过 **few-shot 示例** 和 **function calling 强化** 来优化。
    - **RAG 检索精度**：目前使用 ChromaDB + 句子向量，偶有检索不相关结果；下一步引入 **query 改写**、**重排序（reranking）** 提升准确性。
    - **多实例并发**：当前为单线程，暂不支持高并发；后续将引入异步处理与多进程支持。
    - **记忆持久化**：短期记忆已实现，长期记忆计划采用 **向量数据库 + 摘要** 方式存储。
    """)
st.caption("集成了工具调用（天气、时间、计算、安全命令）和 RAG 知识库（Java 八股文：java基础/集合/IO/锁）")

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 处理用户输入
if prompt := st.chat_input("请输入您的问题"):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 助手响应（流式）
    with st.chat_message("assistant"):
        # 创建一个空的占位符，用于逐步更新
        placeholder = st.empty()
        full_response = ""
        # 调用你的流式生成器
        for chunk in assistant.process_stream(prompt):
            full_response += chunk
            placeholder.markdown(full_response + "▌")  # 显示光标
        placeholder.markdown(full_response)  # 移除光标，显示最终结果

    # 将完整回复存入历史
    st.session_state.messages.append({"role": "assistant", "content": full_response})
