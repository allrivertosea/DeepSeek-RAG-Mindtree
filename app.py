import streamlit as st
import requests
import json
from mindtree_retriever import retrieve_documents
from mindtree_doc_handler import process_documents
from sentence_transformers import CrossEncoder
import torch
import os
from dotenv import load_dotenv, find_dotenv
from logger_config import logger


torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]  # 修复torch类未找到错误
load_dotenv(find_dotenv())  # 加载.env文件内容到应用程序中，使其可通过os.getenv()访问

OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/generate"
MODEL = os.getenv("MODEL", "deepseek-r1:7b") # 确保在 Ollama 中已安装该模型
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text:latest")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

reranker = None  # 初始化交叉编码器（重新排序器）
try:
    reranker = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
except Exception as e:
    st.error(f"加载 CrossEncoder 模型失败: {str(e)}")

# Streamlit配置
st.set_page_config(page_title="Mindtree RAG", layout="wide")      

# 自定义CSS样式
st.markdown("""
    <style>
        /* 全局样式 */
        .stApp { 
            background-color: #f8f9fa;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* 标题样式 */
        h1 { 
            color: #2c3e50;
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 1em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        /* 聊天消息样式 */
        .stChatMessage { 
            border-radius: 15px;
            padding: 15px;
            margin: 15px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .stChatMessage.user { 
            background-color: #e3f2fd;
            border-left: 5px solid #2196f3;
        }
        .stChatMessage.assistant { 
            background-color: #f1f8e9;
            border-left: 5px solid #4caf50;
        }
        
        /* 按钮样式 */
        .stButton>button { 
            background-color: #1976d2;
            color: white;
            border-radius: 25px;
            padding: 10px 20px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #1565c0;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        /* 侧边栏样式 */
        .stSidebar {
            background-color: #ffffff;
            box-shadow: 2px 0 5px rgba(0,0,0,0.1);
        }
        
        /* 文本样式 */
        .stMarkdown {
            color: #333;
        }
        
        /* 滑块样式 */
        .stSlider {
            padding: 2em 0;
        }
        
        /* 复选框样式 */
        .stCheckbox {
            padding: 1em 0;
        }
        
        /* 文件上传区域样式 */
        .stUploader {
            border: 2px dashed #1976d2;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }
        
        /* 进度条样式 */
        .stProgress {
            height: 10px;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)


# 管理会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
    logger.info("Initialized messages session state")
if "retrieval_pipeline" not in st.session_state:
    st.session_state.retrieval_pipeline = None
    logger.info("Initialized retrieval_pipeline session state")
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = True
    logger.info("Initialized rag_enabled session state")
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
    logger.info("Initialized documents_loaded session state")
if "enable_hyde" not in st.session_state:
    st.session_state.enable_hyde = True
    logger.info("Initialized enable_hyde session state")
if "enable_reranking" not in st.session_state:
    st.session_state.enable_reranking = True
    logger.info("Initialized enable_reranking session state")
if "enable_graph_rag" not in st.session_state:
    st.session_state.enable_graph_rag = True
    logger.info("Initialized enable_graph_rag session state")
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.3
    logger.info("Initialized temperature session state")
if "max_contexts" not in st.session_state:
    st.session_state.max_contexts = 3
    logger.info("Initialized max_contexts session state")
if "processing" not in st.session_state:
    st.session_state.processing = False
    logger.info("Initialized processing session state")

# 侧边栏
with st.sidebar:    
    st.header("📁 文档管理")
    uploaded_files = st.file_uploader(
        "上传文档 (PDF/DOCX/TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files and not st.session_state.documents_loaded:
        try:
            with st.spinner("正在处理文档..."):
                logger.info(f"Processing {len(uploaded_files)} uploaded files")
                process_documents(uploaded_files, reranker, EMBEDDINGS_MODEL, OLLAMA_BASE_URL)
                logger.info("Document processing completed successfully")
                st.success("文档处理完成！")
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            st.error(f"文档处理失败: {str(e)}")
    
    st.markdown("---")
    st.header("🔆 RAG设置")
    
    st.session_state.rag_enabled = st.checkbox("启用RAG", value=True)
    st.session_state.enable_hyde = st.checkbox("启用HyDE", value=True)
    st.session_state.enable_reranking = st.checkbox("启用神经重排序", value=True)
    st.session_state.enable_graph_rag = st.checkbox("启用GraphRAG", value=True)
    st.session_state.temperature = st.slider("温度", 0.0, 1.0, 0.3, 0.05)
    st.session_state.max_contexts = st.slider("最大上下文数", 1, 5, 3)
    
    if st.button("清除聊天历史"):
        st.session_state.messages = []
        st.rerun()

    # 页脚（侧边栏右下角）
    st.sidebar.markdown("""
        <div style="position: fixed; bottom: 20px; right: 20px; font-size: 12px; color: #666;">
            <b>由 William 开发</b> &copy; 2025 保留所有权利
        </div>
    """, unsafe_allow_html=True)

# 聊天界面
st.title("🌳 Mindtree RAG")
st.caption("具有GraphRAG、混合检索、神经重排序和聊天历史功能的高级RAG系统")

# 显示消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("请输入你的问题..."):
    try:
        logger.info(f"Received new user prompt: {prompt[:50]}...")
        chat_history = "\n".join([msg["content"] for msg in st.session_state.messages[-5:]])  # 最后5条消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 生成响应
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # 构建上下文
            context = ""
            if st.session_state.rag_enabled and st.session_state.retrieval_pipeline:
                try:
                    logger.info("Retrieving documents for context")
                    docs = retrieve_documents(prompt, OLLAMA_API_URL, MODEL, chat_history)
                    context = "\n".join(
                        f"[来源 {i+1}]: {doc.page_content}" 
                        for i, doc in enumerate(docs)
                    )
                    logger.info(f"Retrieved {len(docs)} documents for context")
                except Exception as e:
                    logger.error(f"Error retrieving documents: {str(e)}")
                    st.error(f"检索错误: {str(e)}")
            
            # 结构化提示词
            system_prompt = f"""使用聊天历史保持上下文：
                聊天历史：
                {chat_history}

                通过以下步骤分析问题和上下文：
                1. 识别关键实体和关系
                2. 检查来源之间的矛盾
                3. 综合多个上下文的信息
                4. 形成结构化响应

                上下文：
                {context}

                问题: {prompt}
                回答:"""
            
            # 流式响应
            logger.info("Generating response using Ollama")
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": MODEL,
                    "prompt": system_prompt,
                    "stream": True,
                    "options": {
                        "temperature": st.session_state.temperature,  # 使用用户选择的动态值
                        "num_ctx": 4096
                    }
                },
                stream=True
            )
            try:
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line.decode())
                        token = data.get("response", "")
                        full_response += token
                        response_placeholder.markdown(full_response + "▌")
                        
                        # 检测到结束标记时停止
                        if data.get("done", False):
                            break
                            
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                logger.info("Response generated successfully")
                
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                st.error(f"生成错误: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": "抱歉，我遇到了一个错误。"})
                
    except Exception as e:
        logger.error(f"Unexpected error in chat processing: {str(e)}")
        st.error("处理请求时发生错误，请重试。")
