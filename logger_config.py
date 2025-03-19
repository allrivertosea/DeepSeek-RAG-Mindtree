import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logger():
    # 创建logs目录
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 设置日志文件名（包含日期）
    log_filename = f'logs/mindtree_rag_{datetime.now().strftime("%Y%m%d")}.log'
    
    # 创建日志记录器
    logger = logging.getLogger('MindtreeRAG')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器（带轮转）
    file_handler = RotatingFileHandler(
        log_filename,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 创建全局日志记录器实例
logger = setup_logger() 