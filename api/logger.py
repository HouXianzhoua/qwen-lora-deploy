# api/logger.py
import logging
import os
from pathlib import Path
from config import LOG_DIR, DEBUG

def setup_logger(name: str = "qwen-lora-api", level: int = None):
    """
    设置并返回一个配置好的 logger。
    
    Args:
        name: logger 的名称，通常使用模块名。
        level: 日志级别。如果为 None，则根据 config.DEBUG 决定。
    
    Returns:
        配置好的 logging.Logger 实例。
    """
    if level is None:
        level = logging.DEBUG if DEBUG else logging.INFO
    
    # 创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加 handler (在模块被多次导入时很重要)
    if logger.handlers:
        return logger
    
    # 创建 formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 创建文件 handler (可选，但强烈推荐)
    log_file_path = LOG_DIR / "app.log"
    try:
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)  # 文件记录 INFO 及以上级别
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # 如果无法创建日志文件，至少确保控制台输出正常
        print(f"⚠️  警告：无法创建日志文件 {log_file_path}，将仅使用控制台输出。错误: {e}")
    
    return logger

# 创建一个全局 logger 实例，供整个项目使用
logger = setup_logger()
