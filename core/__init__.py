"""
核心模块 - 带网络镜像支持
"""
import os
import sys

# 设置 Hugging Face 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 可选：设置超时时间
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from core.agents import build_agents, ModelChoice
from core.knowledge_base import RAGKnowledgeBase
from core.vector_index import FAISSVectorIndex  # 修改：VectorIndex -> FAISSVectorIndex
from core.utils import logger, process_images, ocr_image, classify_issue_type
from core.language_manager import get_language_manager, set_language, Language

__all__ = [
    'build_agents', 'ModelChoice',
    'RAGKnowledgeBase', 'FAISSVectorIndex',  # 修改：添加 FAISSVectorIndex
    'logger', 'process_images', 'ocr_image', 'classify_issue_type',
    'get_language_manager', 'set_language', 'Language'
]