import streamlit as st
import networkx as nx
import re
from logger_config import logger
import time

# 构建知识图谱
# 该函数从文档中提取实体，并在它们之间建立关系，形成知识图谱
def build_knowledge_graph(docs):
    G = nx.Graph()
    try:
        for doc in docs:
            entities = re.findall(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b', doc.page_content)
            if len(entities) > 1:
                for i in range(len(entities) - 1):
                    G.add_edge(entities[i], entities[i + 1])
        
        logger.info(f"构建知识图谱完成，节点数: {len(G.nodes)}, 边数: {len(G.edges)}")
    except Exception as e:
        logger.error(f"知识图谱构建失败: {str(e)}")
    
    return G

def retrieve_from_graph(query, G, top_k=5):
    """
    从知识图谱中检索相关节点
    
    Args:
        query: 查询文本
        G: 知识图谱
        top_k: 返回的节点数量
        
    Returns:
        相关节点列表
    """
    try:
        st.write(f"🔎 正在 GraphRAG 中搜索: {query}")
        start_time = time.time()
        
        # 提取查询中的实体
        query_words = query.lower().split()
        matched_nodes = [node for node in G.nodes if any(word in node.lower() for word in query_words)]
        
        
        if matched_nodes:
            related_nodes = []
            for node in matched_nodes:
                related_nodes.extend(list(G.neighbors(node)))
            
            logger.info(f"GraphRAG 匹配的节点: {matched_nodes}")
            logger.info(f"GraphRAG 检索到的相关节点: {related_nodes[:top_k]}")
            st.write(f"🟢 GraphRAG 匹配的节点: {matched_nodes}")
            st.write(f"🟢 GraphRAG 检索到的相关节点: {related_nodes[:top_k]}")
            return related_nodes[:top_k]
        
        logger.warning(f"GraphRAG 无匹配结果: {query}")
        st.write(f"❌ 没有找到与查询匹配的图谱结果: {query}")
        return []
    except Exception as e:
        logger.error(f"GraphRAG 检索失败: {str(e)}")
        st.error(f"GraphRAG 检索失败: {str(e)}")
        return []
