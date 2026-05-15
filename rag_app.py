import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

# 加载本地环境变量（隐私保护）
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=env_path)

# 页面配置（白底黑字，文字100%可见）
st.set_page_config(
    page_title="RAG智能问答系统",
    layout="wide",
)

# 读取阿里云API配置
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")

# 模型初始化（修复DashScope密钥问题）
llm = ChatOpenAI(model="qwen-plus", temperature=0.1, streaming=True)
embedding = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=api_key
)

# 会话状态初始化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 文档加载器（支持PDF/DOCX/TXT/CSV）
def load_document(file_path: str, file_name: str):
    try:
        if file_name.endswith(".pdf"):
            return PyPDFLoader(file_path).load()
        elif file_name.endswith(".docx"):
            return Docx2txtLoader(file_path).load()
        elif file_name.endswith((".txt", ".md")):
            return TextLoader(file_path, encoding="utf-8").load()
        elif file_name.endswith(".csv"):
            return CSVLoader(file_path, encoding="utf-8").load()
        else:
            st.error(f"不支持的文件格式：{file_name}")
            return []
    except Exception as e:
        st.error(f"文件加载失败：{str(e)}")
        return []

# 构建知识库
def create_knowledge_base(files):
    docs = []
    os.makedirs("uploads", exist_ok=True)

    for file in files:
        path = f"uploads/{file.name}"
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        docs.extend(load_document(path, file.name))

    if not docs:
        return None, 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    db = Chroma.from_documents(splits, embedding, persist_directory="./advanced_db")
    retriever = db.as_retriever(search_kwargs={"k": 5})
    return retriever, len(splits)

# 文档摘要生成
def generate_summary(docs):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "用简洁的中文总结文档核心内容，控制在100字以内"),
        ("user", "总结以下文档：{context}")
    ])
    chain = prompt | llm
    context = "\n".join([doc.page_content for doc in docs[:3]])
    return chain.invoke({"context": context}).content

# 主界面
st.title("RAG智能问答系统")

# 侧边栏知识库管理
with st.sidebar:
    st.header("知识库管理")
    uploaded_files = st.file_uploader(
        "上传文件",
        type=["pdf", "docx", "txt", "md", "csv"],
        accept_multiple_files=True
    )

    col1, col2 = st.columns(2)
    with col1:
        build_btn = st.button("构建知识库", type="primary")
    with col2:
        clear_btn = st.button("清空对话历史")

    if build_btn and uploaded_files:
        with st.spinner("正在构建知识库..."):
            retriever, chunk_num = create_knowledge_base(uploaded_files)
            st.session_state.vector_db = retriever
            st.success(f"构建完成！分块数量：{chunk_num}")
    elif build_btn and not uploaded_files:
        st.warning(" 请先上传文件！")

    if clear_btn:
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# 显示历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户提问
if question := st.chat_input("请输入你的问题..."):
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.chat_history.append(HumanMessage(content=question))
    with st.chat_message("user"):
        st.markdown(question)

with st.chat_message("assistant"):
    placeholder = st.empty()
    full_response = ""

    if not st.session_state.vector_db:
        rresponse = llm.invoke(question)
        full_response = response.content
        placeholder.markdown(full_response)

    else:
        # 修复：用新版invoke方法获取相关文档
        relevant_docs = st.session_state.vector_db.invoke(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是专业的文档问答助手，只能根据提供的文档内容回答，禁止编造。如果文档中没有答案，直接回答：根据文档无法回答该问题。上下文：{context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{question}")
            ])

            chain = prompt | llm
            for chunk in chain.stream({
                "context": context,
                "question": question,
                "chat_history": st.session_state.chat_history
            }):
                full_response += chunk.content
                placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.chat_history.append(AIMessage(content=full_response))

    # 显示答案来源和文档摘要
    if st.session_state.vector_db:
        with st.expander(" 答案来源 & 文档摘要"):
            st.write("**文档摘要：**")
            st.write(generate_summary(relevant_docs))
            st.divider()
            st.write("**相关片段来源：**")
            for idx, doc in enumerate(relevant_docs[:3]):
                st.write(f"来源 {idx + 1}：")
                st.write(doc.page_content[:300] + "...")
                st.divider()
