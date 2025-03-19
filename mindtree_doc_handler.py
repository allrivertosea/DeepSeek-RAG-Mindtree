import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from mindtree_graph import build_knowledge_graph
from rank_bm25 import BM25Okapi
import os
import re
from logger_config import logger
from tqdm import tqdm


# 处理用户上传的文档
# 该函数解析文档内容，将其分割，并存储在不同的检索结构中
def process_documents(uploaded_files, reranker, embedding_model, base_url):
    if st.session_state.documents_loaded:
        logger.info("Documents already loaded, skipping processing")
        return

    st.session_state.processing = True
    documents = []
    
    # 创建临时目录
    if not os.path.exists("temp"):
        os.makedirs("temp")
        logger.info("Created temporary directory")
    
    # 创建进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 处理文件
        for i, file in enumerate(uploaded_files):
            try:
                status_text.text(f"正在处理文件 {i+1}/{len(uploaded_files)}: {file.name}")
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                file_path = os.path.join("temp", file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                logger.info(f"Saved temporary file: {file_path}")

                # 根据文件类型选择加载器
                if file.name.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif file.name.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                elif file.name.endswith(".txt"):
                    loader = TextLoader(file_path)
                else:
                    logger.warning(f"Unsupported file type: {file.name}")
                    continue
                
                # 加载文档
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} pages from {file.name}")
                
                # 清理临时文件
                os.remove(file_path)
                logger.info(f"Removed temporary file: {file_path}")
                
            except Exception as e:
                logger.error(f"Error processing file {file.name}: {str(e)}")
                st.error(f"处理文件 {file.name} 时出错: {str(e)}")
                continue

        if not documents:
            raise ValueError("No documents were successfully processed")

        # 文本分割
        status_text.text("正在分割文本...")
        text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=200, separator="\n")
        texts = text_splitter.split_documents(documents)
        text_contents = [doc.page_content for doc in texts]
        logger.info(f"Split documents into {len(texts)} chunks")

        # 使用Ollama进行嵌入
        status_text.text("正在初始化嵌入模型...")
        embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)
        logger.info("Initialized embeddings")
        
        # 向量存储
        status_text.text("正在创建向量存储...")
        vector_store = FAISS.from_documents(texts, embeddings)
        logger.info("Created and persisted vector store")
        
        # BM25存储
        status_text.text("正在创建BM25索引...")
        bm25_retriever = BM25Retriever.from_texts(
            text_contents, 
            bm25_impl=BM25Okapi,
            preprocess_func=lambda text: re.sub(r"\W+", " ", text).lower().split()
        )
        logger.info("Created BM25 retriever")

        # 组合检索策略 (BM25 + 向量搜索)
        status_text.text("正在配置检索系统...")
        ensemble_retriever = EnsembleRetriever(
            retrievers=[
                bm25_retriever,
                vector_store.as_retriever(search_kwargs={"k": 5})
            ],
            weights=[0.4, 0.6]
        )
        logger.info("Created ensemble retriever")
        # 生成知识图谱
        knowledge_graph = build_knowledge_graph(texts)
        # 将数据存储到会话状态
        st.session_state.retrieval_pipeline = {
            "ensemble": ensemble_retriever,
            "reranker": reranker,
            "texts": text_contents,
            "knowledge_graph": knowledge_graph  # 生成并存储知识图谱
        }

        st.session_state.documents_loaded = True
        st.session_state.processing = False
        
	    # 记录统计信息
        logger.info(f"文档处理完成，总文档数: {len(texts)}")
        logger.info(f"知识图谱节点数: {len(knowledge_graph.nodes)}, 边数: {len(knowledge_graph.edges)}")
    
	    # UI 显示统计信息
        st.success("文档处理完成！")
        st.write(f"📜 处理的文档数: {len(texts)}")
        st.write(f"🔗 知识图谱节点数: {len(knowledge_graph.nodes)}")
        st.write(f"🔗 知识图谱边数: {len(knowledge_graph.edges)}")


    except Exception as e:
        logger.error(f"Error in document processing pipeline: {str(e)}")
        st.error(f"文档处理失败: {str(e)}")
        raise
    finally:
        progress_bar.empty()
        status_text.empty()
