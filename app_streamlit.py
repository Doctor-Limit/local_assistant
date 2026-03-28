import streamlit as st
import time
import json
import os
import logging
from assistant_pkg.assistant import Assistant
from assistant_pkg.user_profile import UserProfile
from assistant_pkg.explainer import ExplanationGenerator
from assistant_pkg.explanation_router import ExplanationRouter
from assistant_pkg.memory import MemoryItem   # 补充导入

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 抑制第三方库的 INFO 日志
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

def format_mbti(mbti_str: str) -> str:
    if len(mbti_str) != 4:
        return mbti_str
    return mbti_str[0] + mbti_str[1] + mbti_str[2] + mbti_str[3]

# 初始化助手（全局单例）
@st.cache_resource
def get_assistant():
    # 在侧边栏创建一个占位符，用于显示和清除提示
    sidebar_placeholder = st.sidebar.empty()
    sidebar_placeholder.info("正在初始化智能体，模块较多，首次加载约需30秒，请稍候...")

    with st.spinner("加载中..."):
        assistant = Assistant()

    # 初始化完成后清除侧边栏提示
    sidebar_placeholder.empty()
    return assistant

assistant = get_assistant()


@st.cache_resource
def get_explainer():
    return ExplanationGenerator(assistant.llm)

# ====== 用户画像初始化 ======
if "user_profile" not in st.session_state:
    st.session_state.user_profile = UserProfile()

if "last_decision_short" not in st.session_state:
    st.session_state.last_decision_short = "暂无"
if "last_decision" not in st.session_state:
    st.session_state.last_decision = "暂无"

if "profile_initialized" not in st.session_state:
    st.session_state.profile_initialized = False

if "history_explanations" not in st.session_state:
    st.session_state.history_explanations = []

# 加载MBTI问卷配置
def load_mbti_questions():
    config_path = os.path.join(os.path.dirname(__file__), "mbti_questions.json")
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {
            "questions": [
                {"text": "1. 你更倾向于：", "options": [
                    {"text": "安静思考", "dimension": "I"},
                    {"text": "积极社交", "dimension": "E"}
                ]},
                {"text": "2. 你更喜欢：", "options": [
                    {"text": "专注于具体事实和细节", "dimension": "S"},
                    {"text": "关注整体模式和可能性", "dimension": "N"}
                ]},
                {"text": "3. 你做决定时更依赖：", "options": [
                    {"text": "逻辑分析", "dimension": "T"},
                    {"text": "个人价值观", "dimension": "F"}
                ]},
                {"text": "4. 你通常：", "options": [
                    {"text": "计划并遵守日程", "dimension": "J"},
                    {"text": "灵活适应变化", "dimension": "P"}
                ]}
            ]
        }

mbti_config = load_mbti_questions()

if not st.session_state.profile_initialized:
    st.info("🎯 为了提供更个性化的解释风格，请先完成简易MBTI测试（仅需几秒）")
    with st.form("mbti_form"):
        st.subheader("选择最符合你的描述：")
        selected_dimensions = []
        for i, q in enumerate(mbti_config["questions"]):
            selected = st.radio(
                q["text"],
                options=[opt["text"] for opt in q["options"]],
                key=f"mbti_q{i}",
                index=0
            )
            for opt in q["options"]:
                if opt["text"] == selected:
                    selected_dimensions.append(opt["dimension"])
                    break
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("确定")
        with col2:
            skipped = st.form_submit_button("跳过")
        if submitted:
            mbti = "".join(selected_dimensions)
            reset_option = st.radio(
                "如何处理已有的认知功能置信度？",
                ["完全重置（使用 MBTI 默认值）", "保留已有置信度（仅更新 MBTI 类型）"],
                key="reset_option"
            )
            if reset_option == "完全重置（使用 MBTI 默认值）":
                initial_functions = UserProfile.init_from_mbti(mbti)
                st.session_state.user_profile.data["cognitive_functions"] = initial_functions
            st.session_state.user_profile.data["mbti"] = mbti
            st.session_state.user_profile.data["last_updated"] = time.time()
            st.session_state.user_profile._save()
            st.session_state.profile_initialized = True
            st.rerun()
        if skipped:
            default_functions = UserProfile.init_from_mbti("INTP")
            st.session_state.user_profile.data["mbti"] = "INTP"
            st.session_state.user_profile.data["cognitive_functions"] = default_functions
            st.session_state.user_profile.data["last_updated"] = time.time()
            st.session_state.user_profile._save()
            st.session_state.profile_initialized = True
            st.rerun()
    st.stop()

# 页面标题和布局
st.set_page_config(page_title="本地智能体助手", page_icon="🤖", layout="wide")
st.title("🤖 本地智能体助手")
st.info(
    "🚀 **演示说明**：\n\n"
    "本智能体当前为提升响应速度与回答质量，使用 **云端 API**（硅基流动 Qwen2.5-7B）进行推理。\n"
    "项目代码完整支持 **本地模型切换**（Ollama + Qwen2.5:1.5B），并可一键回退。\n\n"
    "⚙️ **技术亮点**：模块化架构 · 工具调用 · RAG 检索 · 流式输出 · 对话记忆 · 可解释AI（荣格八维）"
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("石头剪刀布策略"):
        st.session_state.example_prompt = "在“石头剪刀布”游戏中，如果对手总是出石头，你应该出什么？"
        st.rerun()
with col2:
    if st.button("HashMap 线程安全"):
        st.session_state.example_prompt = "为什么 HashMap 线程不安全？"
        st.rerun()
with col3:
    if st.button("电车难题"):
        st.session_state.example_prompt = "一辆有轨电车失控，前方轨道上有5个被绑的人，你可以拉杆让电车转向另一条轨道，但那条轨道上有1个人。你会拉杆吗？为什么？"
        st.rerun()
with col4:
    if st.button("工作选择"):
        st.session_state.example_prompt = "你收到两个工作offer，一个工资高但加班多，一个工资低但工作生活平衡，如何选择？"
        st.rerun()

# 侧边栏
with st.sidebar:
    st.header("📊 系统状态")
    # 指标行：用两行显示四个指标
    col1, col2 = st.columns(2)
    with col1:
        st.metric("😊 情绪", assistant.state.get("mood", "neutral").capitalize())
        st.metric("🧠 短期记忆", len(assistant.memory.short_term))
    with col2:
        kb_size = assistant.retriever.collection.count() if assistant.retriever else "未启用"
        st.metric("📚 知识库", kb_size)
        decision_short = st.session_state.get("last_decision_short", "暂无")
        decision_full = st.session_state.get("last_decision", "暂无")
        st.metric("🎯 上次决策", decision_short, help=decision_full)
    st.divider()
    # 模式选择
    explanation_mode = st.radio(
        "解释模式",
        ["简洁型", "逻辑型", "调试型"],
        index=0,
        key="explanation_mode",
        help="简洁型：仅显示简短总结；逻辑型：显示完整解释、流程图及引用；调试型：显示JSON格式的完整技术细节并允许下载json。"
    )
    # 用户画像
    st.subheader("👤 用户画像")
    profile = st.session_state.user_profile.data
    st.write(f"**MBTI**: {format_mbti(profile.get('mbti', '未知'))}")
    if st.button("🔄 重置用户画像"):
        st.session_state.user_profile.data = {
            "mbti": "INTP",
            "cognitive_functions": UserProfile.init_from_mbti("INTP"),
            "last_updated": time.time()
        }
        st.session_state.user_profile._save()
        st.success("已重置为默认画像。")
        st.rerun()

    # 主要认知功能
    if profile.get("cognitive_functions"):
        top_func = max(profile["cognitive_functions"].items(), key=lambda x: x[1])
        st.write(f"**主要认知功能**: {top_func[0]} ({top_func[1]:.2f})")
    with st.expander("🔍 认知功能置信度详情"):
        if profile.get("cognitive_functions"):
            for func, score in profile["cognitive_functions"].items():
                st.write(f"{func}: {score:.2f}")
                st.progress(score)
    st.caption("这些是我对你的理解，你可以随时修正。")

    # 手动调整认知功能
    with st.expander("✏️ 手动调整认知功能", expanded=False):
        st.write("你可以直接调整我对你的理解强度：")
        current_funcs = st.session_state.user_profile.data.get("cognitive_functions", {})
        new_funcs = {}
        for func, score in sorted(current_funcs.items(), key=lambda x: x[1], reverse=True):
            new_score = st.slider(f"{func}", 0.0, 1.0, score, 0.01, key=f"slider_{func}")
            new_funcs[func] = new_score
        if st.button("保存修改"):
            st.session_state.user_profile.data["cognitive_functions"] = new_funcs
            st.session_state.user_profile.data["last_updated"] = time.time()
            st.session_state.user_profile._save()
            st.success("画像已更新，下次对话将优先使用你设定的风格。")
            st.rerun()

    # 反馈记录与校准
    with st.expander("📊 反馈与校准"):
        logs = st.session_state.user_profile._load_feedback_log()
        if logs:
            st.subheader("最近反馈记录")
            for entry in logs[-5:][::-1]:
                time_str = time.strftime("%m-%d %H:%M", time.localtime(entry["timestamp"]))
                st.write(f"**{time_str}** {entry['function']} - {entry['feedback']}")
                st.caption(f"Q: {entry['question']}")
        if st.button("📊 查看交互历史摘要"):
            stats = st.session_state.user_profile.get_recent_feedback_stats(recent_n=20)
            if stats:
                st.subheader("最近20次反馈统计")
                for func, counts in stats.items():
                    st.write(f"**{func}**:  👍 {counts['positive']}  👎 {counts['negative']}")
            else:
                st.info("暂无反馈记录，请通过解释面板的评分按钮提供反馈。")
        if st.button("🔄 基于近期反馈重新校准画像"):
            st.session_state.user_profile.recalibrate_from_feedback(recent_n=20)
            st.success("已根据最近20次反馈重新校准认知功能置信度。")
            st.rerun()

    st.divider()

    # 记忆库（短期 + 长期搜索）
    st.subheader("🧠 记忆库")
    memory_items = list(assistant.memory.short_term)

    # 短期记忆展示
    if memory_items:
        with st.expander("📋 短期记忆", expanded=False):
            reversed_items = list(reversed(memory_items))[-10:]  # 显示最近10条
            for idx, item in enumerate(reversed_items):
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    short_content = item.content[:60] + "..." if len(item.content) > 60 else item.content
                    st.markdown(f"**{item.role}** ({item.source_type}) - {short_content}")
                    st.caption(f"ID: `{item.mem_id}`  |  置信度: {item.confidence:.2f}")
                with col2:
                    if st.button("查看", key=f"view_short_{item.mem_id}"):
                        st.session_state.selected_mem_id = item.mem_id
                        st.session_state.show_mem_detail = True
                        st.rerun()
    else:
        st.info("暂无短期记忆，开始对话后记录")


    # 长期记忆检索
    with st.expander("🔍 长期记忆检索", expanded=False):
        search_term = st.text_input("输入关键词搜索长期记忆", key="long_term_search_input")
        if st.button("搜索", key="search_long_term"):
            if search_term and assistant.memory.long_term:
                with st.spinner("检索中..."):
                    results = assistant.memory.long_term.search(search_term, top_k=10)
                    st.session_state.long_term_search_results = results
            else:
                st.warning("请输入关键词或长期记忆未启用")
        if "long_term_search_results" in st.session_state and st.session_state.long_term_search_results:
            for item in st.session_state.long_term_search_results:
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    short_content = item.content[:60] + "..." if len(item.content) > 60 else item.content
                    st.markdown(f"**{item.role}** ({item.source_type}) - {short_content}")
                    st.caption(f"ID: `{item.mem_id}`  |  置信度: {item.confidence:.2f}")
                with col2:
                    if st.button("查看", key=f"view_long_{item.mem_id}"):
                        st.session_state.selected_mem_id = item.mem_id
                        st.session_state.show_mem_detail = True
                        st.rerun()
    # 记忆详情区域（显示选中记忆的完整信息）
    if st.session_state.get("show_mem_detail", False) and "selected_mem_id" in st.session_state:
        mem_id = st.session_state.selected_mem_id
        target_mem = None
        # 短期查找
        for item in memory_items:
            if item.mem_id == mem_id:
                target_mem = item
                break
        # 长期查找（如果短期没找到）
        if not target_mem and "long_term_search_results" in st.session_state:
            for item in st.session_state.long_term_search_results:
                if item.mem_id == mem_id:
                    target_mem = item
                    break
        if target_mem:
            with st.container():
                st.subheader("🔍 记忆详情")
                st.write(f"**ID**: {target_mem.mem_id}")
                st.write(f"**角色**: {target_mem.role}")
                st.write(f"**类型**: {target_mem.source_type}")
                st.write(f"**时间**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(target_mem.timestamp))}")
                st.write(f"**置信度**: {target_mem.confidence:.2f}")
                st.write("**内容**:")
                st.write(target_mem.content)
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ 正确", key=f"verify_correct_{mem_id}"):
                        # 增加置信度
                        new_conf = min(1.0, target_mem.confidence + 0.05)
                        # 更新短期或长期记忆
                        if target_mem in assistant.memory.short_term:
                            target_mem.confidence = new_conf
                        elif "long_term_search_results" in st.session_state:
                            # 更新长期记忆列表中的对象
                            for idx, it in enumerate(st.session_state.long_term_search_results):
                                if it.mem_id == mem_id:
                                    st.session_state.long_term_search_results[idx].confidence = new_conf
                                    break
                            if assistant.memory.long_term:
                                assistant.memory.long_term.collection.update(
                                    ids=[target_mem.mem_id],
                                    metadatas=[{"confidence": new_conf}]
                                )
                        st.success(f"置信度已更新为 {new_conf:.2f}")
                        st.rerun()
                with col2:
                    if st.button("❌ 错误", key=f"verify_wrong_{mem_id}"):
                        # 降低置信度
                        new_conf = max(0.0, target_mem.confidence - 0.05)
                        if target_mem in assistant.memory.short_term:
                            target_mem.confidence = new_conf
                        elif "long_term_search_results" in st.session_state:
                            for idx, it in enumerate(st.session_state.long_term_search_results):
                                if it.mem_id == mem_id:
                                    st.session_state.long_term_search_results[idx].confidence = new_conf
                                    break
                            if assistant.memory.long_term:
                                assistant.memory.long_term.collection.update(
                                    ids=[target_mem.mem_id],
                                    metadatas=[{"confidence": new_conf}]
                                )
                        st.warning(f"置信度已降低为 {new_conf:.2f}")
                        st.rerun()
                if st.button("关闭"):
                    st.session_state.show_mem_detail = False
                    st.rerun()
        else:
            st.error("未找到记忆")

# 主界面其他内容（保持不变）
with st.expander("📝 已知优化项 & 下一步计划"):
    st.markdown("""
    - **可解释AI**：已实现基于荣格八维认知功能的解释生成，根据用户画像选择解释风格，差异化风格体现需要优化。
    - **复杂语句工具调用**：当前在用户意图模糊或多步推理时，模型可能无法正确触发工具；
    - **多实例并发**：当前为单线程，暂不支持高并发；后续将引入异步处理与多进程支持。
    - **记忆持久化**：短期记忆与长期记忆尝试采用记忆模型存储。
    - **机械解释实现**：不支持查看模型具体想法权重占比。
    - **迎合性问题**：长时间对话模型可能存在迎合性选择，尝试引入元认知进行自回答。
    - **针对性反馈**：目前未对反馈进行操作，只有日志记录。
    """)
st.caption("集成了工具调用（天气、时间、计算、安全命令）和 RAG 知识库（Java 八股文：java基础/集合/IO/锁）")

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息及交互按钮
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 如果是助手消息且有反馈元数据，添加按钮和解释区域
        if msg["role"] == "assistant" and "feedback_meta" in msg:
            meta = msg["feedback_meta"]
            msg_key = f"msg_{idx}"

            # ---- 反馈按钮区（仅当未反馈时显示） ----
            if not msg.get("feedback_done", False):
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    if st.button("👍 有帮助", key=f"like_{msg_key}"):
                        st.session_state.user_profile.update_function_confidence(
                            function=meta["chosen_func"],
                            delta=0.05,
                            context="用户对解释风格给予正面反馈",
                            question=meta["question"],
                            answer=meta["answer"],
                            old_confidence=meta["chosen_confidence"]
                        )
                        msg["feedback_done"] = True
                        st.success("感谢反馈！")
                        st.rerun()
                with col2:
                    if st.button("👎 没帮助", key=f"dislike_{msg_key}"):
                        st.session_state.user_profile.update_function_confidence(
                            function=meta["chosen_func"],
                            delta=-0.03,
                            context="用户对解释风格给予负面反馈",
                            question=meta["question"],
                            answer=meta["answer"],
                            old_confidence=meta["chosen_confidence"]
                        )
                        msg["feedback_done"] = True
                        st.success("感谢反馈！")
                        st.rerun()
                with col3:
                    if st.button("🤔 思考过程", key=f"think_{msg_key}"):
                        with st.spinner("正在生成思考过程..."):
                            explainer = get_explainer()
                            # 直接生成推理链（不经过路由和缓存）
                            reasoning_chain = explainer.generate_reasoning_chain(meta["answer"], meta.get("ctx", {}))
                            mermaid_reasoning = explainer.tree_to_mermaid(reasoning_chain)
                            with st.popover("思考过程", use_container_width=True):
                                if mermaid_reasoning:
                                    st.markdown(f"```mermaid\n{mermaid_reasoning}\n```")
                                else:
                                    # 如果没有流程图，尝试显示纯文本推理链
                                    chain_text = reasoning_chain.get("root", "") + "\n" + "\n".join(
                                        [c["text"] for c in reasoning_chain.get("children", [])])
                                    if chain_text.strip():
                                        st.markdown(chain_text)
                                    else:
                                        st.warning("思考过程生成失败，请重试。")
                with col4:
                    # ---- 显示解释（如果存在且需要显示） 直接在点击后打开一个窗口展示
                    if st.button("📖 查看解释", key=f"explain_{msg_key}"):
                        with st.popover("解释详情"):
                            # 如果解释尚未生成，则立即生成
                            if msg.get("explanation") is None:
                                with st.spinner("正在生成解释..."):
                                    explainer = get_explainer()
                                    router = ExplanationRouter(st.session_state.user_profile, explainer)
                                    history = st.session_state.history_explanations[
                                        -5:] if st.session_state.history_explanations else []
                                    _, viz = router.route(
                                        "knowledge", meta["answer"], meta.get("ctx", {}),
                                        history_explanations=history
                                    )
                                    msg["explanation"] = viz

                            # 显示解释内容
                            viz = msg["explanation"]
                            mode = st.session_state.get("explanation_mode", "逻辑型")

                            #专注于展示逻辑树
                            if viz.get("mermaid"):
                                st.markdown("**推理流程图**")
                                st.markdown(f"```mermaid\n{viz['mermaid']}\n```")
                            if viz.get("mermaid_reasoning"):
                                st.markdown("**符号推理链**")
                                st.markdown(f"```mermaid\n{viz['mermaid_reasoning']}\n```")

                                # 其他内容全部折叠
                            with st.expander("📖 完整解释文本"):
                                if mode == "简洁型":
                                    st.markdown(viz.get("reasoning_chain", "")[:200] + "...")
                                else:
                                    st.markdown(viz.get("reasoning_chain", ""))

                            if viz.get("citations_used"):
                                with st.expander("📚 引用来源"):
                                    for cit in viz["citations_used"]:
                                        st.markdown(f"> {cit.get('content', '')}")

                            if viz.get("tool_calls"):
                                with st.expander("🔧 工具调用记录"):
                                    for tc in viz["tool_calls"]:
                                        st.markdown(f"- {tc['name']}({tc['arguments']}) → {tc['result']}")

                            # ---- 对比其他解释风格（在解释详情内部） ----
                            all_funcs = list(viz.get("all_functions_confidence", {}).keys())
                            current_func = meta["chosen_func"]
                            other_funcs = [f for f in all_funcs if f != current_func]
                            if other_funcs:
                                with st.expander("🔍 对比其他解释风格"):
                                    selected_other = st.selectbox("选择其他认知功能", other_funcs,
                                                                  key=f"compare_{msg_key}")
                                    if st.button("生成对比解释", key=f"compare_btn_{msg_key}"):
                                        with st.spinner(f"生成 {selected_other} 风格解释..."):
                                            explainer = get_explainer()
                                            other_data = explainer.explain(selected_other, meta["answer"],
                                                                           meta.get("ctx", {}))
                                            other_text = other_data.get("text", "")
                                            other_mermaid = other_data.get("mermaid")
                                            st.markdown(other_text)
                                            if other_mermaid:
                                                st.markdown("**推理流程图**")
                                                st.markdown(f"```mermaid\n{other_mermaid}\n```")
                                            # 对比评分按钮
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                if st.button("👍 更喜欢这个", key=f"prefer_{selected_other}_{msg_key}"):
                                                    old_conf = st.session_state.user_profile.data[
                                                        "cognitive_functions"].get(
                                                        selected_other, 0.5)
                                                    st.session_state.user_profile.update_function_confidence(
                                                        function=selected_other,
                                                        delta=0.05,
                                                        context="用户更喜欢对比解释",
                                                        question=meta["question"],
                                                        answer=meta["answer"],
                                                        old_confidence=old_conf
                                                    )
                                                    st.success(f"已更新偏好，{selected_other} 置信度提升")
                                                    st.rerun()
                                            with col2:
                                                if st.button("👎 还是原来的好", key=f"original_prefer_{msg_key}"):
                                                    old_conf = st.session_state.user_profile.data[
                                                        "cognitive_functions"].get(
                                                        current_func, 0.5)
                                                    st.session_state.user_profile.update_function_confidence(
                                                        function=current_func,
                                                        delta=0.03,
                                                        context="用户更偏好原解释风格",
                                                        question=meta["question"],
                                                        answer=meta["answer"],
                                                        old_confidence=old_conf
                                                    )
                                                    st.success("感谢反馈！")
                                                    st.rerun()

                            # ---- 重新生成解释按钮 ----
                            if st.button("🔄 重新生成解释", key=f"regenerate_popup_{msg_key}"):
                                with st.spinner("重新生成中..."):
                                    explainer = get_explainer()
                                    router = ExplanationRouter(st.session_state.user_profile, explainer)
                                    _, new_viz = router.route("knowledge", meta["answer"], meta.get("ctx", {}))
                                    msg["explanation"] = new_viz
                                    st.rerun()
                with col5:
                    if st.button("🔄 重新生成", key=f"regenerate_answer_{msg_key}"):
                        with st.spinner("正在重新生成回答..."):
                            new_full_response = ""
                            for chunk in assistant.process_stream(meta["question"]):
                                new_full_response += chunk
                            # 更新回答内容和元数据
                            msg["content"] = new_full_response
                            msg["feedback_meta"]["answer"] = new_full_response
                            msg["feedback_meta"]["ctx"] = assistant.last_context
                            # 清除旧解释数据，确保下次查看解释时重新生成
                            msg["explanation"] = None
                            msg["show_explanation"] = False
                            msg["feedback_done"] = False
                        st.success("回答已重新生成！")
                        st.rerun()
            else:
                # 已反馈，显示一个标记
                st.caption("✅ 已反馈")

# 处理用户输入（支持示例和直接输入）
if "example_prompt" in st.session_state and st.session_state.example_prompt:
    prompt = st.session_state.example_prompt
    del st.session_state.example_prompt
elif prompt := st.chat_input("请输入您的问题"):
    pass
else:
    prompt = None

# 处理用户输入
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    task_type = "knowledge"
    chosen_func, chosen_confidence = st.session_state.user_profile.get_preferred_function(task_type)
    st.session_state.last_decision_short = chosen_func
    st.session_state.last_decision = f"{chosen_func} (置信度: {chosen_confidence:.2f})"

    with st.chat_message("assistant"):
        style_msg = f"我将用 **{chosen_func}** 风格回答，置信度 **{chosen_confidence:.2f}**。"
        st.markdown(style_msg)
        placeholder = st.empty()
        full_response = ""
        for chunk in assistant.process_stream(prompt):
            full_response += chunk
            placeholder.markdown(style_msg + "\n\n" + full_response + "▌")
        placeholder.markdown(style_msg + "\n\n" + full_response)

    # 将助手消息存入历史，附带反馈元数据，但解释字段为空
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "feedback_meta": {
            "chosen_func": chosen_func,
            "chosen_confidence": chosen_confidence,
            "question": prompt,
            "answer": full_response,
            "ctx": assistant.last_context
        },
        "feedback_done": False,
        "explanation": None,
        "show_explanation": False
    })

    # 强制刷新页面，确保按钮立即显示
    st.rerun()
