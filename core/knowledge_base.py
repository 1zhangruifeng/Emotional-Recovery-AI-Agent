"""
RAG知识库管理 - 支持增量累加和多种数据源
"""

from pathlib import Path
from typing import List, Dict, Optional
import json
import hashlib

from core.vector_index import FAISSVectorIndex


class RAGKnowledgeBase:
    """RAG知识库管理器 - 支持增量累加"""

    def __init__(self, index_path: str = "./data/knowledge_base/psychology_index"):
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self.index = FAISSVectorIndex(dimension=384, use_ivf=False)
        # 延迟导入，避免循环依赖和阻塞
        self._dataset_loader = None
        self._crawler = None

        self.is_ready = False

    @property
    def dataset_loader(self):
        """延迟加载 dataset_loader"""
        if self._dataset_loader is None:
            from core.dataset_loader import DatasetLoader
            self._dataset_loader = DatasetLoader()
        return self._dataset_loader

    @property
    def crawler(self):
        """延迟加载 crawler"""
        if self._crawler is None:
            from core.crawler import PsychologyCrawler
            self._crawler = PsychologyCrawler()
        return self._crawler

    def load(self) -> bool:
        """加载已有的FAISS数据库"""
        success = self.index.load(str(self.index_path))
        if success:
            self.is_ready = True
            stats = self.index.get_stats()
            print(f"✅ 知识库加载成功！共 {stats['total_entries']} 条知识")
        else:
            self.is_ready = False
            print("⚠️ 未找到知识库文件，将创建新知识库")
        return success

    def incremental_add(self, new_knowledge: List[Dict], show_progress: bool = True) -> Dict:
        """
        增量添加新知识（自动去重）

        Args:
            new_knowledge: 新知识列表
            show_progress: 是否显示进度

        Returns:
            添加结果统计
        """
        if not self.is_ready:
            # 如果还没有索引，先创建空索引
            self.index.build_index([], show_progress=False)
            self.is_ready = True

        # 直接调用 add_incremental，它会处理创建索引的逻辑
        result = self.index.add_incremental(new_knowledge, show_progress)

        # 保存更新后的索引
        if result['added'] > 0:
            self.index.save(str(self.index_path))

        return result

    def build_from_sources(self,
                          use_datasets: bool = True,
                          use_crawler: bool = False,
                          use_builtin: bool = False,
                          crawler_keywords: List[str] = None,
                          incremental: bool = True) -> Dict:
        """
        从多个源构建/更新知识库

        Args:
            use_datasets: 是否使用公开数据集
            use_crawler: 是否使用爬虫采集
            use_builtin: 是否使用内置知识
            crawler_keywords: 爬虫关键词列表
            incremental: 是否增量模式（True=累加，False=覆盖）

        Returns:
            添加结果统计
        """
        all_new_knowledge = []

        # 1. 公开数据集
        if use_datasets:
            print("\n📚 加载公开数据集...")
            dataset_knowledge = self._load_available_datasets()
            all_new_knowledge.extend(dataset_knowledge)
            print(f"   📊 数据集: {len(dataset_knowledge)} 条")

        # 2. 爬虫采集
        if use_crawler:
            print("\n🕷️ 爬取网络资源...")
            crawler_knowledge = self._run_crawler(crawler_keywords)
            all_new_knowledge.extend(crawler_knowledge)
            print(f"   📊 爬虫: {len(crawler_knowledge)} 条")

        # 3. 内置知识（兜底）
        if use_builtin:
            print("\n📖 加载内置知识...")
            builtin_knowledge = self._load_builtin_knowledge()
            all_new_knowledge.extend(builtin_knowledge)
            print(f"   📊 内置: {len(builtin_knowledge)} 条")

        if not all_new_knowledge:
            print("⚠️ 没有找到任何新知识")
            return {'total': 0, 'added': 0, 'duplicates': 0}

        # 4. 增量添加
        print(f"\n📈 {'增量累加' if incremental else '覆盖重建'}模式...")

        if incremental:
            # 加载现有索引
            self.load()
            # 增量添加
            result = self.incremental_add(all_new_knowledge, show_progress=True)
        else:
            # 覆盖模式
            self.index.build_index(all_new_knowledge, show_progress=True)
            self.index.save(str(self.index_path))
            self.is_ready = True
            result = {
                'total': len(all_new_knowledge),
                'added': len(all_new_knowledge),
                'duplicates': 0,
                'total_after': len(all_new_knowledge)
            }

        # 显示统计
        print("\n" + "=" * 50)
        print("📊 构建结果:")
        print(f"   📥 输入: {result['total']} 条")
        print(f"   ✅ 新增: {result['added']} 条")
        print(f"   🗑️ 重复: {result['duplicates']} 条")
        print(f"   📚 知识库总计: {result.get('total_after', self.index.get_stats()['total_entries'])} 条")
        print("=" * 50)

        return result

    def _load_available_datasets(self) -> List[Dict]:
        """加载所有可用的公开数据集（支持 PDF、JSON、CSV、TXT、JSONL、MD）"""
        knowledge = []
        datasets_dir = Path("./data/datasets")
        datasets_dir.mkdir(parents=True, exist_ok=True)

        # PDF文件
        for pdf_file in datasets_dir.glob("*.pdf"):
            print(f"   📑 加载PDF: {pdf_file.name}")
            knowledge.extend(self._load_pdf_file(pdf_file))

        # JSON文件
        for json_file in datasets_dir.glob("*.json"):
            print(f"   📄 加载: {json_file.name}")
            knowledge.extend(self._load_json_file(json_file))

        # JSONL文件
        for jsonl_file in datasets_dir.glob("*.jsonl"):
            print(f"   📄 加载: {jsonl_file.name}")
            knowledge.extend(self._load_jsonl_file(jsonl_file))

        # CSV文件
        for csv_file in datasets_dir.glob("*.csv"):
            print(f"   📄 加载: {csv_file.name}")
            knowledge.extend(self._load_csv_file(csv_file))

        # TXT文件
        for txt_file in datasets_dir.glob("*.txt"):
            print(f"   📄 加载: {txt_file.name}")
            knowledge.extend(self._load_txt_file(txt_file))

        # MD文件（Markdown）
        for md_file in datasets_dir.glob("*.md"):
            print(f"   📄 加载: {md_file.name}")
            knowledge.extend(self._load_txt_file(md_file))

        return knowledge

    def _load_pdf_file(self, file_path: Path) -> List[Dict]:
        """加载PDF文件，提取文本内容"""
        knowledge = []
        try:
            # 尝试使用 pdfplumber（推荐）
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    full_text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            full_text += page_text + "\n"
            except ImportError:
                # 备选：使用 PyPDF2
                try:
                    from PyPDF2 import PdfReader
                    reader = PdfReader(file_path)
                    full_text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            full_text += page_text + "\n"
                except ImportError:
                    print(f"   ⚠️ 请安装PDF解析库: pip install pdfplumber")
                    return knowledge

            if not full_text or len(full_text) < 100:
                print(f"   ⚠️ PDF内容过短或无法提取: {file_path.name}")
                return knowledge

            # 按段落分割
            chunks = self._split_text_into_chunks(full_text, file_path.stem)

            for chunk in chunks:
                if len(chunk['content']) > 100:
                    knowledge.append({
                        'title': chunk['title'],
                        'content': chunk['content'],
                        'source': f"pdf_{file_path.stem}",
                        'issue_type': self._classify_content(chunk['content']),
                        'type': 'pdf'
                    })

            print(f"   ✅ 成功提取 {len(knowledge)} 条知识")

        except Exception as e:
            print(f"   ❌ PDF加载失败 {file_path.name}: {e}")

        return knowledge

    def _split_text_into_chunks(self, text: str, base_title: str, max_chunk_size: int = 1500) -> List[Dict]:
        """将长文本分割成多个块"""
        chunks = []

        # 按双换行分割段落
        paragraphs = text.split('\n\n')

        current_chunk = ""
        chunk_index = 1

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 如果当前块加上新段落会超过限制，则保存当前块
            if len(current_chunk) + len(para) > max_chunk_size and current_chunk:
                chunks.append({
                    'title': f"{base_title}_part{chunk_index}",
                    'content': current_chunk.strip()
                })
                chunk_index += 1
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # 保存最后一个块
        if current_chunk:
            chunks.append({
                'title': f"{base_title}_part{chunk_index}",
                'content': current_chunk.strip()
            })

        return chunks

    def _load_json_file(self, file_path: Path) -> List[Dict]:
        """加载JSON文件"""
        knowledge = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict) and 'items' in data:
                    items = data['items']
                else:
                    items = [data] if isinstance(data, dict) else []

                for item in items:
                    content = item.get('content', item.get('text', ''))
                    if content and len(content) > 50:
                        knowledge.append({
                            'title': item.get('title', item.get('name', file_path.stem)),
                            'content': content,
                            'source': f"dataset_{file_path.stem}",
                            'issue_type': self._classify_content(content),
                            'type': 'dataset'
                        })
        except Exception as e:
            print(f"   ❌ JSON加载失败: {e}")
        return knowledge

    def _load_jsonl_file(self, file_path: Path) -> List[Dict]:
        """加载JSONL文件"""
        knowledge = []
        try:
            import jsonlines
            with jsonlines.open(file_path) as reader:
                for item in reader:
                    content = item.get('content', item.get('text', ''))
                    if content and len(content) > 50:
                        knowledge.append({
                            'title': item.get('title', item.get('name', file_path.stem)),
                            'content': content,
                            'source': f"dataset_{file_path.stem}",
                            'issue_type': self._classify_content(content),
                            'type': 'dataset'
                        })
        except ImportError:
            # 如果没有jsonlines，手动逐行解析
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            item = json.loads(line)
                            content = item.get('content', item.get('text', ''))
                            if content and len(content) > 50:
                                knowledge.append({
                                    'title': item.get('title', item.get('name', file_path.stem)),
                                    'content': content,
                                    'source': f"dataset_{file_path.stem}",
                                    'issue_type': self._classify_content(content),
                                    'type': 'dataset'
                                })
            except Exception as e:
                print(f"   ❌ JSONL加载失败: {e}")
        except Exception as e:
            print(f"   ❌ JSONL加载失败: {e}")
        return knowledge

    def _load_csv_file(self, file_path: Path) -> List[Dict]:
        """加载CSV文件"""
        knowledge = []
        try:
            import csv
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 尝试找内容列
                    content = row.get('content') or row.get('text') or row.get('article') or ''
                    if not content:
                        # 取第一列作为内容
                        content = list(row.values())[0] if row else ''

                    if content and len(content) > 50:
                        knowledge.append({
                            'title': row.get('title', row.get('name', file_path.stem)),
                            'content': content,
                            'source': f"dataset_{file_path.stem}",
                            'issue_type': self._classify_content(content),
                            'type': 'dataset'
                        })
        except Exception as e:
            print(f"   ❌ CSV加载失败: {e}")
        return knowledge

    def _load_txt_file(self, file_path: Path) -> List[Dict]:
        """加载TXT文件（按段落分割）"""
        knowledge = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 按双换行分割段落
            paragraphs = content.split('\n\n')
            for i, para in enumerate(paragraphs):
                para = para.strip()
                if len(para) > 100:
                    knowledge.append({
                        'title': f"{file_path.stem}_{i+1}",
                        'content': para,
                        'source': f"dataset_{file_path.stem}",
                        'issue_type': self._classify_content(para),
                        'type': 'dataset'
                    })
        except Exception as e:
            print(f"   ❌ TXT加载失败: {e}")
        return knowledge

    def _run_crawler(self, keywords: List[str] = None) -> List[Dict]:
        """运行爬虫采集数据"""
        knowledge = []

        if keywords is None:
            keywords = [
                "情感支持", "共情技巧", "认知行为疗法", "情绪管理",
                "人际关系修复", "焦虑缓解", "自我关怀", "心理韧性",
                "分手心理", "失恋恢复", "压力管理", "正念冥想"
            ]

        # 爬取百度百科
        print("   📖 爬取百度百科...")
        baike_results = self.crawler.crawl_baike_psychology(keywords[:10])
        for item in baike_results:
            knowledge.append({
                'title': item['title'],
                'content': item['content'],
                'source': item['source'],
                'issue_type': self._classify_content(item['content']),
                'type': 'crawled',
                'quality_score': item.get('quality_score', 0.5)
            })

        # 爬取知乎（快速模式，避免太慢）
        print("   📝 爬取知乎话题...")
        zhihu_results = self.crawler.crawl_zhihu_topic(pages=1)
        for item in zhihu_results:
            knowledge.append({
                'title': item['title'],
                'content': item['content'],
                'source': item['source'],
                'issue_type': self._classify_content(item['content']),
                'type': 'crawled',
                'quality_score': item.get('quality_score', 0.5)
            })

        return knowledge

    def _load_builtin_knowledge(self) -> List[Dict]:
        """加载内置知识库"""
        builtin_knowledge = [
            {"title": "有效共情的四个步骤",
             "content": "有效共情包含四个关键步骤：1. 倾听而不评判，让对方充分表达；2. 识别并命名情绪，如'听起来你感到很沮丧'；3. 验证情绪的合理性，让对方知道'有这种感觉是完全正常的'；4. 表达理解和支持。共情的核心是让对方感到被看见、被理解。",
             "source": "builtin", "issue_type": "general", "type": "builtin"},

            {"title": "认知行为疗法 - 识别思维扭曲",
             "content": "常见的思维扭曲类型：1. 非黑即白：把事情看成全好或全坏；2. 过度概括：把一次失败看成永远会失败；3. 灾难化：总是预期最坏的结果；4. 个人化：把所有问题都归咎于自己；5. 读心术：自以为知道别人在想什么。识别这些思维模式是认知重构的第一步。",
             "source": "builtin", "issue_type": "general", "type": "builtin"},

            {"title": "焦虑缓解的接地技巧",
             "content": "5-4-3-2-1接地技巧：说出你能看到的5样东西；触摸你能摸到的4样东西；注意你能听到的3种声音；闻到你周围的2种气味；说出你能尝到的1种味道。这个技巧能帮助你将注意力从焦虑转移到当下，迅速降低焦虑水平。",
             "source": "builtin", "issue_type": "mental health", "type": "builtin"},

            {"title": "分手后情绪恢复指南",
             "content": "分手后的情绪恢复是一个过程：1. 允许自己感受悲伤、愤怒等情绪，不要压抑；2. 建立新的日常规律，保持生活节奏；3. 重新发现自己的身份和价值，找回丢失的兴趣爱好；4. 与朋友和家人保持联系，不孤立自己；5. 给自己时间，不要急于开始新的恋情。",
             "source": "builtin", "issue_type": "romantic breakup", "type": "builtin"},

            {"title": "工作压力管理策略",
             "content": "职场压力管理实用策略：1. 设置明确的工作边界，学会说'不'；2. 分解大任务为小步骤，降低焦虑感；3. 定期进行短暂休息，使用番茄工作法；4. 建立支持系统，与信任的同事交流；5. 区分可控和不可控因素，专注于能改变的事情。",
             "source": "builtin", "issue_type": "workplace stress", "type": "builtin"},

            {"title": "人际关系冲突解决技巧",
             "content": "解决人际冲突的有效方法：1. 使用'我'语句表达感受，避免指责对方；2. 积极倾听，先理解对方的立场再表达自己；3. 寻找共同目标，而不是争论谁对谁错；4. 给彼此冷静的时间，避免在情绪激动时沟通；5. 关注解决方案而不是追究责任。",
             "source": "builtin", "issue_type": "interpersonal conflict", "type": "builtin"},

            {"title": "家庭沟通改善技巧",
             "content": "改善家庭沟通的方法：1. 选择合适的沟通时机，避免在情绪激动或疲惫时沟通；2. 表达感受而非指责，使用'我感到...'的句式；3. 尝试理解对方的出发点和立场；4. 建立家庭会议制度，定期交流；5. 必要时寻求专业的家庭治疗师帮助。",
             "source": "builtin", "issue_type": "family issues", "type": "builtin"},
        ]
        return builtin_knowledge

    def _classify_content(self, text: str) -> str:
        """对内容进行分类"""
        text_lower = text.lower()

        categories = {
            'romantic breakup': ['分手', '失恋', '前任', '离婚', 'breakup', 'heartbreak'],
            'interpersonal conflict': ['吵架', '冲突', '矛盾', 'conflict', 'argument'],
            'workplace stress': ['工作', '职场', '压力', 'work', 'career', 'stress'],
            'mental health': ['焦虑', '抑郁', '情绪', 'anxiety', 'depression'],
            'family issues': ['家庭', '父母', '家人', 'family', 'parent'],
            'academic anxiety': ['考试', '学习', '学业', 'exam', 'study']
        }

        for category, keywords in categories.items():
            if any(kw in text_lower for kw in keywords):
                return category
        return 'general'

    def search(self, query: str, issue_type: str = None, k: int = 3) -> List[Dict]:
        """搜索相关知识"""
        if not self.is_ready:
            return []
        return self.index.search(query, k=k, issue_type=issue_type, min_score=0.15)

    def add_knowledge(self, title: str, content: str, source: str = "manual",
                      issue_type: str = "general"):
        """动态添加单条知识"""
        if not self.is_ready:
            self.load()

        new_item = {
            'title': title,
            'content': content,
            'source': source,
            'issue_type': issue_type,
            'type': 'manual'
        }

        result = self.incremental_add([new_item], show_progress=False)
        return result

    def get_stats(self) -> Dict:
        """获取知识库统计信息"""
        if self.index:
            return self.index.get_stats()
        return {'total_entries': 0, 'is_ready': self.is_ready}

    def get_all_knowledge(self) -> List[Dict]:
        """获取所有知识（用于查看）"""
        if self.index:
            return self.index.get_all_knowledge()
        return []