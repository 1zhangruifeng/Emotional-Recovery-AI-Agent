#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查看知识库统计信息
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.knowledge_base import RAGKnowledgeBase


def main():
    print("\n" + "=" * 60)
    print("   📊 知识库统计工具")
    print("=" * 60)

    rag = RAGKnowledgeBase()

    print("\n🔍 正在加载知识库...")
    print(f"   索引路径: {rag.index_path}")

    # 检查文件是否存在
    faiss_path = rag.index_path.with_suffix('.faiss')
    pkl_path = rag.index_path.with_suffix('.pkl')

    print(f"   FAISS文件: {faiss_path} ({faiss_path.exists() and '存在' or '不存在'})")
    print(f"   PKL文件: {pkl_path} ({pkl_path.exists() and '存在' or '不存在'})")

    success = rag.load()

    if not success:
        print("\n❌ 知识库不存在或加载失败！")
        print("\n请先构建知识库:")
        print("   1. 运行: python scripts/download_hf_datasets.py")
        print("   2. 运行: python scripts/build_knowledge_base.py")
        print("   3. 选择选项 2 加载本地数据集")
        return

    stats = rag.get_stats()
    print(f"\n📈 索引统计:")
    print(f"   总条目数: {stats['total_entries']}")
    print(f"   索引类型: {stats['index_type']}")
    print(f"   向量维度: {stats['dimension']}")

    # 按来源分类统计
    if rag.index and rag.index.knowledge_base:
        source_stats = {}
        issue_stats = {}

        for item in rag.index.knowledge_base:
            source = item.get('source', 'unknown')
            source_stats[source] = source_stats.get(source, 0) + 1

            issue = item.get('issue_type', 'general')
            issue_stats[issue] = issue_stats.get(issue, 0) + 1

        if source_stats:
            print(f"\n📂 按来源分布:")
            for source, count in sorted(source_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"   - {source}: {count} 条")
        else:
            print(f"\n📂 按来源分布: 无数据")

        if issue_stats:
            print(f"\n🏷️ 按问题类型分布:")
            for issue, count in sorted(issue_stats.items(), key=lambda x: x[1], reverse=True):
                print(f"   - {issue}: {count} 条")
        else:
            print(f"\n🏷️ 按问题类型分布: 无数据")

        # 显示示例
        print(f"\n📖 知识示例（前3条）:")
        for i, item in enumerate(rag.index.knowledge_base[:3]):
            print(f"\n   [{i + 1}] {item.get('title', '无标题')[:50]}")
            print(f"       来源: {item.get('source', '未知')}")
            print(f"       类型: {item.get('issue_type', 'general')}")
            print(f"       内容预览: {item.get('content', '')[:100]}...")
    else:
        print("\n⚠️ 知识库中没有数据")


if __name__ == "__main__":
    main()