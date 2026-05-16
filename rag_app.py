import os
import streamlit as st

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

# =========================
# 加载环境变量
# =========================

load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY")

if not api_key:
    st.error("未检测到 DASHSCOPE_API_KEY，请检查 .env 文件")
    st.stop()

# =========================
# 设置环境变量
# =========================

os.environ["OPENAI_API_KEY"] = api_key
os.environ["DASHSCOPE_API_KEY"] = api_key
os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# =========================
# 页面配置
# =========================

st.set_page_config(
    page_title="RAG智能问答系统",
    layout="wide"
)

st.title("RAG智能问答系统")

# =========================
# 初始化模型
# =========================

llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0.1,
    streaming=True
)

embedding = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=api_key
)

# =========================
# Session 初始化
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

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
            loader = TextLoader(file_path, encoding="utf-8")

        elif file_name.endswith(".csv"):
            loader = CSVLoader(file_path, encoding="utf-8")

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

        docs.extend(load_document(file_path, file.name))

    if not docs:
        return None, 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = splitter.split_documents(docs)

    db = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory="./advanced_db"
    )

    retriever = db.as_retriever(
        search_kwargs={"k": 5}
    )

    return retriever, len(splits)

# =========================
# 文档摘要
# =========================

def generate_summary(docs):

    try:

        context = "\n".join([
            doc.page_content
            for doc in docs[:3]
        ])

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "请用简洁中文总结文档内容，100字以内"
            ),
            (
                "user",
                "{context}"
            )
        ])

        chain = prompt | llm

        response = chain.invoke({
            "context": context
        })

        return response.content

    except Exception as e:
        return f"摘要生成失败：{str(e)}"

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

                retriever, chunk_num = create_knowledge_base(uploaded_files)

                st.session_state.vector_db = retriever

                st.success(f"知识库构建完成，共 {chunk_num} 个分块")

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

    st.session_state.chat_history.append(
        HumanMessage(content=question)
    )

    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(question)

    # AI 回复
    with st.chat_message("assistant"):

        placeholder = st.empty()

        full_response = ""

        # =========================
        # 普通聊天模式
        # =========================

        if not st.session_state.vector_db:

            try:

                response = llm.invoke(question)

                full_response = response.content

                placeholder.markdown(full_response)

            except Exception as e:

                st.error(f"模型调用失败：{str(e)}")

        # =========================
        # RAG模式
        # =========================

        else:

            try:

                relevant_docs = st.session_state.vector_db.invoke(question)

                context = "\n\n".join([
                    doc.page_content
                    for doc in relevant_docs
                ])

                prompt = ChatPromptTemplate.from_messages([

                    (
                        "system",
                        """
你是专业的文档问答助手。

你只能根据提供的文档内容回答问题。

禁止编造不存在的信息。

文档内容如下：

{context}
"""
                    ),

                    MessagesPlaceholder(
                        variable_name="chat_history"
                    ),

                    (
                        "user",
                        "{question}"
                    )

                ])

                chain = prompt | llm

                for chunk in chain.stream({

                    "context": context,
                    "question": question,
                    "chat_history": st.session_state.chat_history

                }):

                    if chunk.content:

                        full_response += chunk.content

                        placeholder.markdown(
                            full_response + "▌"
                        )

                placeholder.markdown(full_response)

                # 保存AI消息
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })

                st.session_state.chat_history.append(
                    AIMessage(content=full_response)
                )

                # =========================
                # 来源展示
                # =========================

                with st.expander("答案来源 & 文档摘要"):

                    st.write("### 文档摘要")

                    summary = generate_summary(relevant_docs)

                    st.write(summary)

                    st.divider()

                    st.write("### 相关文档片段")

                    for idx, doc in enumerate(relevant_docs[:3]):

                        st.write(f"来源 {idx + 1}")

                        st.write(
                            doc.page_content[:300] + "..."
                        )

                        st.divider()

            except Exception as e:

                st.error(f"RAG检索失败：{str(e)}")
