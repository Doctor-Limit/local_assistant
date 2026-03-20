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
st.caption("集成了工具调用（天气、时间、计算、安全命令）和 RAG 知识库（Java 八股文）")

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
