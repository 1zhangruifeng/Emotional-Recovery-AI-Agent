"""
从 Hugging Face 下载真实的中文情感/心理数据集
"""

import os
import json
from pathlib import Path
from typing import List, Dict

# 设置 Hugging Face 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def download_chnsenticorp():
    """
    下载 ChnSentiCorp 中文情感分析数据集
    来源: https://huggingface.co/datasets/seamew/ChnSentiCorp
    包含: 7000+ 条中文评论情感标注（积极/消极）
    """
    try:
        from datasets import load_dataset

        print("📚 正在下载 ChnSentiCorp (中文情感分析数据集)...")

        dataset = load_dataset("seamew/ChnSentiCorp", split="train")

        knowledge_items = []

        for i, item in enumerate(dataset):
            text = item.get('text', '')
            label = item.get('label', '')

            if not text or len(text) < 50:
                continue

            # 情感标签转换
            if label == 1:
                emotion_label = "积极情感"
            elif label == 0:
                emotion_label = "消极情感"
            else:
                emotion_label = "未分类"

            content = f"[情感标签: {emotion_label}]\n{text}"

            knowledge_items.append({
                'title': f"情感分析_{i+1}",
                'content': content,
                'source': 'ChnSentiCorp',
                'issue_type': 'sentiment_analysis',
                'type': 'sentiment'
            })

            if len(knowledge_items) >= 1000:
                break

        print(f"   ✅ 下载完成，共 {len(knowledge_items)} 条")
        return knowledge_items

    except Exception as e:
        print(f"   ❌ ChnSentiCorp 下载失败: {e}")
        return []


def download_lccc():
    """
    下载 LCCC 中文心理咨询对话数据集
    来源: https://huggingface.co/datasets/lccc
    包含: 大量中文心理咨询对话
    """
    try:
        from datasets import load_dataset

        print("📚 正在下载 LCCC (中文心理咨询对话数据集)...")

        # LCCC 有多个配置，尝试加载
        knowledge_items = []

        # 尝试加载训练集
        try:
            dataset = load_dataset("lccc", split="train", streaming=True)

            for i, item in enumerate(dataset):
                dialogue = item.get('dialogue', '')
                if dialogue and len(dialogue) > 100:
                    knowledge_items.append({
                        'title': f"心理对话_{len(knowledge_items)+1}",
                        'content': dialogue,
                        'source': 'LCCC',
                        'issue_type': classify_by_dialogue(dialogue),
                        'type': 'dialogue'
                    })

                    if len(knowledge_items) >= 800:
                        break

        except Exception as e:
            print(f"   ⚠️ 训练集加载失败: {e}")

        print(f"   ✅ 下载完成，共 {len(knowledge_items)} 条")
        return knowledge_items

    except Exception as e:
        print(f"   ❌ LCCC 下载失败: {e}")
        return []


def download_cmdd():
    """
    下载 CMDD 中文心理健康对话数据集
    """
    try:
        from datasets import load_dataset

        print("📚 正在下载 CMDD (中文心理健康对话)...")

        knowledge_items = []

        try:
            dataset = load_dataset("cmdd", split="train", streaming=True)

            for i, item in enumerate(dataset):
                dialogue = item.get('dialogue', '')
                if dialogue and len(dialogue) > 100:
                    knowledge_items.append({
                        'title': f"心理健康对话_{len(knowledge_items)+1}",
                        'content': dialogue,
                        'source': 'CMDD',
                        'issue_type': classify_by_dialogue(dialogue),
                        'type': 'dialogue'
                    })

                    if len(knowledge_items) >= 500:
                        break

        except:
            pass

        print(f"   ✅ 下载完成，共 {len(knowledge_items)} 条")
        return knowledge_items

    except Exception as e:
        print(f"   ❌ CMDD 下载失败: {e}")
        return []


def download_daily_dialog():
    """
    下载 DailyDialog 中文情感对话数据集
    """
    try:
        from datasets import load_dataset

        print("📚 正在下载 DailyDialog (日常对话情感数据集)...")

        knowledge_items = []

        try:
            dataset = load_dataset("daily_dialog", split="train", streaming=True)

            for i, item in enumerate(dataset):
                dialogue = item.get('dialog', [])
                emotion = item.get('emotion', [])

                if dialogue and len(dialogue) > 0:
                    dialogue_text = " ".join(dialogue)
                    if len(dialogue_text) > 100:
                        emotion_text = f"[情感标签: {emotion}]" if emotion else ""
                        content = f"{emotion_text}\n{dialogue_text}"

                        knowledge_items.append({
                            'title': f"日常对话_{len(knowledge_items)+1}",
                            'content': content,
                            'source': 'DailyDialog',
                            'issue_type': 'general',
                            'type': 'dialogue'
                        })

                        if len(knowledge_items) >= 500:
                            break

        except:
            pass

        print(f"   ✅ 下载完成，共 {len(knowledge_items)} 条")
        return knowledge_items

    except Exception as e:
        print(f"   ❌ DailyDialog 下载失败: {e}")
        return []


def download_go_emotions_chinese():
    """
    下载 GoEmotions 的中文翻译版本或相关数据
    """
    try:
        from datasets import load_dataset

        print("📚 正在下载 GoEmotions (英文情感数据集，可翻译使用)...")

        knowledge_items = []

        try:
            dataset = load_dataset("go_emotions", split="train", streaming=True)

            for i, item in enumerate(dataset):
                text = item.get('text', '')
                labels = item.get('labels', [])

                if text and len(text) > 50:
                    # 情感标签映射
                    emotion_map = {
                        0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance',
                        4: 'approval', 5: 'caring', 6: 'confusion', 7: 'curiosity',
                        8: 'desire', 9: 'disappointment', 10: 'disapproval', 11: 'disgust',
                        12: 'embarrassment', 13: 'excitement', 14: 'fear', 15: 'gratitude',
                        16: 'grief', 17: 'joy', 18: 'love', 19: 'nervousness',
                        20: 'optimism', 21: 'pride', 22: 'realization', 23: 'relief',
                        24: 'remorse', 25: 'sadness', 26: 'surprise', 27: 'neutral'
                    }

                    emotion_names = [emotion_map.get(l, 'unknown') for l in labels if l in emotion_map]

                    content = f"[情感标签: {', '.join(emotion_names[:3])}]\n{text}"

                    knowledge_items.append({
                        'title': f"情感表达_{len(knowledge_items)+1}",
                        'content': content,
                        'source': 'GoEmotions',
                        'issue_type': 'sentiment_analysis',
                        'type': 'emotion'
                    })

                    if len(knowledge_items) >= 800:
                        break

        except Exception as e:
            print(f"   ⚠️ 加载失败: {e}")

        print(f"   ✅ 下载完成，共 {len(knowledge_items)} 条")
        return knowledge_items

    except Exception as e:
        print(f"   ❌ GoEmotions 下载失败: {e}")
        return []


def download_psych8k_original():
    """
    从 GitHub 下载 Psych8K 原始数据集
    """
    import requests

    print("📚 正在从 GitHub 下载 Psych8K...")

    urls = [
        "https://raw.githubusercontent.com/qiuchang/Psy8K/main/data/train.json",
        "https://raw.githubusercontent.com/qiuchang/Psy8K/main/data/valid.json",
    ]

    knowledge_items = []

    for url in urls:
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()

                for i, item in enumerate(data):
                    dialogue = item.get('dialogue', '')
                    if dialogue and len(dialogue) > 100:
                        knowledge_items.append({
                            'title': f"Psych8K对话_{len(knowledge_items)+1}",
                            'content': dialogue,
                            'source': 'Psych8K',
                            'issue_type': classify_by_dialogue(dialogue),
                            'type': 'dialogue'
                        })

                        if len(knowledge_items) >= 1000:
                            break

                print(f"   ✅ 从 GitHub 获取 {len(knowledge_items)} 条")
                break

        except Exception as e:
            print(f"   ⚠️ 从 {url} 下载失败: {e}")
            continue

    return knowledge_items


def classify_by_dialogue(dialogue: str) -> str:
    """根据对话内容分类"""
    text = dialogue.lower()

    if any(k in text for k in ['分手', '失恋', '前任', '离婚', '感情']):
        return 'romantic breakup'
    elif any(k in text for k in ['吵架', '冲突', '矛盾', '争吵', '人际']):
        return 'interpersonal conflict'
    elif any(k in text for k in ['工作', '职场', '加班', '老板', '同事']):
        return 'workplace stress'
    elif any(k in text for k in ['焦虑', '抑郁', '失眠', '恐慌', '压力']):
        return 'mental health'
    elif any(k in text for k in ['家庭', '父母', '家人', '亲子', '婚姻']):
        return 'family issues'
    elif any(k in text for k in ['考试', '学习', '学业', '成绩']):
        return 'academic anxiety'
    else:
        return 'general'


def save_to_datasets(knowledge_items: List[Dict], filename: str):
    """保存到 datasets 目录"""
    if not knowledge_items:
        return 0

    output_path = Path(f"./data/datasets/{filename}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 如果文件已存在，加载现有数据
    existing_items = []
    if output_path.exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_items = json.load(f)
        except:
            existing_items = []

    # 合并去重
    existing_contents = {item['content'][:200] for item in existing_items}
    new_items = [item for item in knowledge_items if item['content'][:200] not in existing_contents]

    all_items = existing_items + new_items

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)

    print(f"   💾 保存到 {filename}: {len(new_items)} 条新知识 (总计 {len(all_items)} 条)")

    return len(new_items)


def main():
    print("\n" + "=" * 60)
    print("   🌐 下载真实的中文情感/心理数据集")
    print("=" * 60)
    print()

    all_knowledge = []

    # 1. ChnSentiCorp - 中文情感分析
    print("\n" + "-" * 40)
    data1 = download_chnsenticorp()
    if data1:
        all_knowledge.extend(data1)
        save_to_datasets(data1, "chnsenticorp.json")

    # 2. LCCC - 心理咨询对话
    print("\n" + "-" * 40)
    data2 = download_lccc()
    if data2:
        all_knowledge.extend(data2)
        save_to_datasets(data2, "lccc_dialogue.json")

    # 3. Psych8K - GitHub 源
    print("\n" + "-" * 40)
    data3 = download_psych8k_original()
    if data3:
        all_knowledge.extend(data3)
        save_to_datasets(data3, "psych8k.json")

    # 4. CMDD - 心理健康对话
    print("\n" + "-" * 40)
    data4 = download_cmdd()
    if data4:
        all_knowledge.extend(data4)
        save_to_datasets(data4, "cmdd.json")

    # 5. GoEmotions - 英文情感（可翻译）
    print("\n" + "-" * 40)
    data5 = download_go_emotions_chinese()
    if data5:
        all_knowledge.extend(data5)
        save_to_datasets(data5, "goemotions.json")

    # 6. DailyDialog
    print("\n" + "-" * 40)
    data6 = download_daily_dialog()
    if data6:
        all_knowledge.extend(data6)
        save_to_datasets(data6, "dailydialog.json")

    # 统计
    print("\n" + "=" * 60)
    print("📊 下载统计:")
    print(f"   总计获取: {len(all_knowledge)} 条真实数据集")

    if all_knowledge:
        source_counts = {}
        for item in all_knowledge:
            source = item.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1

        print("\n   按来源分布:")
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {source}: {count} 条")

        type_counts = {}
        for item in all_knowledge:
            issue_type = item.get('issue_type', 'general')
            type_counts[issue_type] = type_counts.get(issue_type, 0) + 1

        print("\n   按问题类型分布:")
        for issue_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {issue_type}: {count} 条")

    print("\n" + "=" * 60)
    print("✅ 下载完成！")
    print("   现在运行: python scripts/build_knowledge_base.py")
    print("   选择选项 2 加载本地数据集")
    print("=" * 60)


if __name__ == "__main__":
    main()