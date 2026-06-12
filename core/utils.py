import logging
import tempfile
from pathlib import Path
import pytesseract
from PIL import Image
import cv2
import numpy as np

# 日志配置
logger = logging.getLogger("emotional_recovery")
logger.setLevel(logging.ERROR)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)


def process_images(files) -> list:
    """将上传文件保存到临时路径，返回本地图片路径列表。"""
    processed = []
    for file in files:
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.name}")
            tmp.write(file.getvalue())
            tmp.close()
            processed.append({"filepath": Path(tmp.name)})
        except Exception as e:
            logger.error(f"Error processing image {file.name}: {e}")
    return processed


def ocr_image(file) -> str:
    """对图片进行 OCR 文字识别"""
    try:
        img = Image.open(file)
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 8
        )
        h, w = binary.shape
        binary = cv2.resize(binary, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        file.seek(0)
        return pytesseract.image_to_string(binary, lang="chi_sim+eng")
    except Exception as e:
        logger.error(f"OCR错误: {e}")
        return ""


def classify_issue_type(text: str) -> str:
    """对用户输入的问题进行分类"""
    text_lower = text.lower() if text else ""

    positive_keywords = [
        '开心', '高兴', '快乐', '很棒', '幸福', '兴奋', '满意', '开心了',
        'happy', 'glad', 'joy', 'excited', 'great', 'good mood'
    ]
    if any(keyword in text_lower for keyword in positive_keywords):
        return 'positive mood'

    categories = {
        'romantic breakup': ['分手', '失恋', '前任', 'ex', '离婚', 'breakup', 'heartbreak'],
        'interpersonal conflict': ['吵架', '争吵', '冲突', '矛盾', '绝交', 'fight', 'argument', 'conflict'],
        'workplace stress': ['工作', '职场', '老板', '同事', '绩效', '加班', 'work', 'job', 'career'],
        'mental health': ['焦虑', '抑郁', '压力', '失眠', '情绪', 'anxiety', 'depression', 'stress'],
        'family issues': ['家人', '家庭', '父母', '亲戚', 'family', 'parents'],
        'financial stress': ['钱', '经济', '贫穷', '债务', 'money', 'debt'],
        'academic anxiety': ['考试', '挂科', '学习', '学业', '论文', 'exam', 'study', 'academic']
    }

    for category, keywords in categories.items():
        if any(kw in text_lower for kw in keywords):
            return category

    return 'general emotional distress'
