#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建知识库脚本 - 支持增量累加和自动去重

运行方式:
    python scripts/build_knowledge_base.py

功能:
    - 选项1: 爬虫采集 + 更新到FAISS数据库
    - 选项2: 加载本地数据集 + 更新到FAISS数据库
    - 自动去重，只添加新知识
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.knowledge_base import RAGKnowledgeBase


def print_banner():
    print()
    print("=" * 60)
    print("   🌱 情感恢复AI助手 - 知识库构建工具")
    print("=" * 60)
    print()
    print("📌 说明:")
    print("   - 第一次运行：创建新知识库")
    print("   - 后续运行：增量添加新知识（自动去重）")
    print()


def main():
    print_banner()

    # 检查数据目录
    datasets_dir = Path("./data/datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # 显示已有知识库状态
    rag = RAGKnowledgeBase()
    existing_count = 0
    if rag.load():
        existing_count = rag.get_stats()['total_entries']
        print(f"\n📊 现有知识库: {existing_count} 条知识")
    else:
        print(f"\n📊 现有知识库: 不存在（将创建新知识库）")

    print()
    print("=" * 60)
    print("请选择操作:")
    print("   [1] 爬虫采集 - 从网络爬取心理学知识并更新到数据库")
    print("   [2] 本地数据集 - 加载 data/datasets/ 目录下的文件并更新到数据库")
    print("=" * 60)

    choice = input("\n请选择 (1 或 2): ").strip()

    if choice not in ["1", "2"]:
        print("❌ 无效选项，请输入 1 或 2")
        return

    start_time = time.time()

    # 在选项1中，修改爬虫调用方式
    if choice == "1":
        print("\n" + "=" * 60)
        print("🕷️ 选项1: 爬虫采集（百度百科）")
        print("=" * 60)
        print("   来源: 百度百科心理学词条")
        print("   词条数量: 30个")
        print("   预计耗时: 1-2分钟")
        print()

        confirm = input("确认开始爬虫? (y/n，默认y): ").strip().lower()
        if confirm == 'n':
            print("已取消")
            return

        # 直接使用爬虫采集，不经过 build_from_sources
        from core.crawler import PsychologyCrawler, convert_crawled_to_knowledge

        crawler = PsychologyCrawler()
        crawled_results = crawler.crawl_all_sources()

        if crawled_results:
            crawler_knowledge = convert_crawled_to_knowledge(crawled_results)
            print(f"\n📊 爬虫获取: {len(crawler_knowledge)} 条知识")

            # 增量添加到知识库
            result = rag.incremental_add(crawler_knowledge, show_progress=True)
            print(f"\n📈 添加结果: 新增 {result['added']} 条, 重复 {result['duplicates']} 条")
        else:
            print("\n⚠️ 没有获取到任何知识")
            result = {'total': 0, 'added': 0, 'duplicates': 0}

    elif choice == "2":
        # 选项2：本地数据集
        print("\n" + "=" * 60)
        print("📂 选项2: 本地数据集")
        print("=" * 60)

        # 显示数据集目录内容
        dataset_files = list(datasets_dir.glob("*"))
        if dataset_files:
            print("   发现以下文件:")
            for f in dataset_files:
                file_size = f.stat().st_size / 1024  # KB
                print(f"   - {f.name} ({file_size:.1f} KB)")
        else:
            print("   ⚠️ data/datasets/ 目录为空")
            print("   请将以下格式的文件放入该目录:")
            print("   - PDF, JSON, CSV, TXT, JSONL, MD")

        print()
        confirm = input("确认加载这些文件? (y/n，默认y): ").strip().lower()
        if confirm == 'n':
            print("已取消")
            return

        result = rag.build_from_sources(
            use_datasets=True,  # 加载本地数据集
            use_crawler=False,  # 不使用爬虫
            use_builtin=False,  # 不使用内置知识（避免重复）
            incremental=True  # 增量模式
        )

    elapsed_time = time.time() - start_time

    # 最终统计
    stats = rag.get_stats()
    print()
    print("=" * 60)
    print("📊 最终统计:")
    print(f"   ⏱️ 耗时: {elapsed_time:.1f} 秒")
    print(f"   📥 输入: {result['total']} 条")
    print(f"   ✅ 新增: {result['added']} 条")
    print(f"   🗑️ 重复: {result['duplicates']} 条")
    print(f"   📚 知识库总计: {stats['total_entries']} 条")
    print(f"   📁 索引位置: ./data/knowledge_base/")
    print("=" * 60)
    print("\n✅ 完成！现在可以运行: python main.py")
    print()


if __name__ == "__main__":
    main()