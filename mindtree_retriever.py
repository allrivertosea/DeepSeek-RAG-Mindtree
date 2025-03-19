import streamlit as st
from mindtree_graph import retrieve_from_graph
from langchain_core.documents import Document
import requests
import json
from logger_config import logger


# 使用HyDE进行查询扩展
def expand_query(query,uri,model):
    try:
        response = requests.post(uri, json={
            "model": model,
            "prompt": f"为以下问题生成一个假设性答案: {query}",
            "stream": False
        }).json()
        return f"{query}\n{response.get('response', '')}"
    except Exception as e:
        st.error(f"查询扩展失败: {str(e)}")
        return query


# 高级检索管道
# 该函数使用 BM25、FAISS 和 GraphRAG 来检索相关文档，并支持神经重新排序

def retrieve_documents(query, uri, model, chat_history=""):
    expanded_query = expand_query(f"{chat_history}\n{query}", uri, model) if st.session_state.enable_hyde else query
    
    # 使用 BM25 + FAISS 进行文档检索
    with st.spinner("正在检索相关文档..."):
        docs = st.session_state.retrieval_pipeline["ensemble"].invoke(expanded_query)
        logger.info(f"BM25 + FAISS 检索到 {len(docs)} 篇相关文档")

    # GraphRAG 相关性检索
    if st.session_state.enable_graph_rag:
        graph_results = retrieve_from_graph(query, st.session_state.retrieval_pipeline["knowledge_graph"])
        
        # 显示调试信息
        logger.info(f"GraphRAG 检索到 {len(graph_results)} 个相关节点")
        st.write(f"🔍 GraphRAG 检索到的节点: {graph_results}")

        # 确保 GraphRAG 结果的格式正确
        graph_docs = [Document(page_content=node) for node in graph_results]
        # 如果 GraphRAG 结果有效，则合并到检索结果中
        if graph_docs:
            docs = graph_docs + docs  # 合并 GraphRAG 结果与 BM25 + FAISS 结果

    # 神经重新排序 (如果启用)
    if st.session_state.enable_reranking:
        pairs = [[query, doc.page_content] for doc in docs]  # 重新排序时使用 `page_content`
        scores = st.session_state.retrieval_pipeline["reranker"].predict(pairs)

        # 按照重新排序的得分对文档进行排序
        ranked_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
    else:
        ranked_docs = docs

    return ranked_docs[:st.session_state.max_contexts]  # 根据最大上下文数返回结果