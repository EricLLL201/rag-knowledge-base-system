import os
import streamlit as st

from dotenv import load_dotenv

from langchain_community.chat_models import ChatOllama

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

# =========================
# 加载环境变量
# =========================

load_dotenv()

# =========================
# 页面配置
# =========================

st.set_page_config(
    page_title="RAG智能问答系统",
    layout="wide"
)

st.title("RAG智能问答系统")

# =========================
# 初始化本地AI模型
# =========================

llm = ChatOllama(
    model="qwen:7b",
    temperature=0.1
)

# =========================
# Session 初始化
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "docs" not in st.session_state:
    st.session_state.docs = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================
# 文件读取
# =========================

def load_document(file_path, file_name):

    try:

        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)

        elif file_name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)

        elif file_name.endswith(".txt") or file_name.endswith(".md"):
            loader = TextLoader(
                file_path,
                encoding="utf-8"
            )

        elif file_name.endswith(".csv"):
            loader = CSVLoader(
                file_path,
                encoding="utf-8"
            )

        else:
            st.error(f"不支持的文件格式：{file_name}")
            return []

        return loader.load()

    except Exception as e:
        st.error(f"文件读取失败：{str(e)}")
        return []

# =========================
# 构建知识库
# =========================

def create_knowledge_base(files):

    docs = []

    os.makedirs("uploads", exist_ok=True)

    for file in files:

        file_path = f"uploads/{file.name}"

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        docs.extend(
            load_document(
                file_path,
                file.name
            )
        )

    if not docs:
        return [], 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = splitter.split_documents(docs)

    return splits, len(splits)

# =========================
# 简单检索
# =========================

def retrieve_docs(question, docs):

    results = []

    question = question.lower()

    for doc in docs:

        content = doc.page_content.lower()

        if any(
            word in content
            for word in question.split()
        ):
            results.append(doc)

    return results[:3]

# =========================
# 文档摘要
# =========================

def generate_summary(docs):

    try:

        text = "\n".join([
            doc.page_content
            for doc in docs[:3]
        ])

        prompt = f"""
请用100字总结以下文档：

{text}
"""

        response = llm.invoke(prompt)

        return response.content

    except Exception as e:
        return str(e)

# =========================
# 侧边栏
# =========================

with st.sidebar:

    st.header("知识库管理")

    uploaded_files = st.file_uploader(
        "上传文件",
        type=["pdf", "docx", "txt", "md", "csv"],
        accept_multiple_files=True
    )

    col1, col2 = st.columns(2)

    with col1:

        build_btn = st.button(
            "构建知识库",
            type="primary"
        )

    with col2:

        clear_btn = st.button(
            "清空对话"
        )

    # 构建知识库
    if build_btn:

        if not uploaded_files:

            st.warning("请先上传文件")

        else:

            with st.spinner("正在构建知识库..."):

                docs, chunk_num = create_knowledge_base(
                    uploaded_files
                )

                st.session_state.docs = docs

                st.success(
                    f"知识库构建完成，共 {chunk_num} 个分块"
                )

    # 清空历史
    if clear_btn:

        st.session_state.messages = []
        st.session_state.chat_history = []

        st.rerun()

# =========================
# 显示历史消息
# =========================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

# =========================
# 用户输入
# =========================

question = st.chat_input("请输入你的问题...")

if question:

    # 保存用户消息
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)

    # AI 回复
    with st.chat_message("assistant"):

        placeholder = st.empty()

        full_response = ""

        try:

            # 必须先上传文件
            if not st.session_state.docs:

                st.error("请先上传文件并构建知识库")

                st.stop()

            # =========================
            # RAG模式
            # =========================

            relevant_docs = retrieve_docs(
                question,
                st.session_state.docs
            )

            # 没找到相关内容
            if not relevant_docs:

                full_response = "知识库中未找到相关信息"

                placeholder.markdown(full_response)

            else:

                context = "\n\n".join([
                    doc.page_content
                    for doc in relevant_docs
                ])

                prompt = f"""
你是专业的文档问答助手。

你只能根据知识库内容回答问题。

禁止编造不存在的信息。

如果知识库没有答案，
请明确回答：
“知识库中未找到相关信息”。

知识库内容：

{context}

用户问题：

{question}
"""

                response = llm.invoke(prompt)

                full_response = response.content

                placeholder.markdown(full_response)

                # 来源展示
                with st.expander("答案来源 & 文档摘要"):

                    st.write("### 文档摘要")

                    summary = generate_summary(
                        relevant_docs
                    )

                    st.write(summary)

                    st.divider()

                    st.write("### 相关文档片段")

                    for idx, doc in enumerate(relevant_docs):

                        st.write(f"来源 {idx + 1}")

                        st.write(
                            doc.page_content[:300] + "..."
                        )

                        st.divider()

            # 保存AI消息
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })

        except Exception as e:

            st.error(f"模型调用失败：{str(e)}")
