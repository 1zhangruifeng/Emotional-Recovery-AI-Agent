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
            if len(crawler_knowledge) < 5:
                if incremental and self._has_builtin_knowledge():
                    print("   ⚠️ 网络爬取较少，已有内置兜底知识，本次不重复追加")
                else:
                    print("   ⚠️ 网络爬取较少，自动加入增强内置心理知识作为兜底")
                    fallback_knowledge = self._load_builtin_knowledge()
                    all_new_knowledge.extend(fallback_knowledge)
                    print(f"   📊 兜底知识: {len(fallback_knowledge)} 条")

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

    def _has_builtin_knowledge(self) -> bool:
        if not self.is_ready:
            self.load()
        if not self.index:
            return False
        return any(item.get("source") == "builtin" for item in self.index.knowledge_base)

    def _load_available_datasets(self) -> List[Dict]:
        """加载所有可用的公开数据集（支持 PDF、JSON、CSV、TXT、JSONL、MD）"""
        knowledge = []
        datasets_dir = Path("./data/datasets")
        datasets_dir.mkdir(parents=True, exist_ok=True)

        # PDF文件
        for pdf_file in datasets_dir.rglob("*.pdf"):
            print(f"   📑 加载PDF: {pdf_file.name}")
            knowledge.extend(self._load_pdf_file(pdf_file))

        # JSON文件
        for json_file in datasets_dir.rglob("*.json"):
            print(f"   📄 加载: {json_file.name}")
            knowledge.extend(self._load_json_file(json_file))

        # JSONL文件
        for jsonl_file in datasets_dir.rglob("*.jsonl"):
            print(f"   📄 加载: {jsonl_file.name}")
            knowledge.extend(self._load_jsonl_file(jsonl_file))

        # CSV文件
        for csv_file in datasets_dir.rglob("*.csv"):
            print(f"   📄 加载: {csv_file.name}")
            knowledge.extend(self._load_csv_file(csv_file))

        # TXT文件
        for txt_file in datasets_dir.rglob("*.txt"):
            print(f"   📄 加载: {txt_file.name}")
            knowledge.extend(self._load_txt_file(txt_file))

        # MD文件（Markdown）
        for md_file in datasets_dir.rglob("*.md"):
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

    def _split_text_into_chunks(self, text: str, base_title: str, max_chunk_size: int = 900) -> List[Dict]:
        """将长文本分割成多个块"""
        chunks = []
        import re

        text = re.sub(r'\s+', ' ', text).strip()
        paragraphs = re.split(r'(?<=[。！？.!?])\s+', text)
        if len(paragraphs) <= 1:
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
                current_chunk = current_chunk[-120:] + para
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
                    content = self._extract_item_content(item)
                    if content and len(content) > 50:
                        knowledge.append({
                            'title': item.get('title', item.get('name', file_path.stem)),
                            'content': content,
                            'source': f"dataset_{file_path.stem}",
                            'url': item.get('url', ''),
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
                    content = self._extract_item_content(item)
                    if content and len(content) > 50:
                        knowledge.append({
                            'title': item.get('title', item.get('name', file_path.stem)),
                            'content': content,
                            'source': f"dataset_{file_path.stem}",
                            'url': item.get('url', ''),
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
                            content = self._extract_item_content(item)
                            if content and len(content) > 50:
                                knowledge.append({
                                    'title': item.get('title', item.get('name', file_path.stem)),
                                    'content': content,
                                    'source': f"dataset_{file_path.stem}",
                                    'url': item.get('url', ''),
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
                    content = (
                        row.get('content') or row.get('text') or row.get('article') or
                        row.get('body') or row.get('answer') or row.get('response') or
                        row.get('question') or ''
                    )
                    if not content:
                        # 取第一列作为内容
                        content = list(row.values())[0] if row else ''

                    if content and len(content) > 50:
                        knowledge.append({
                            'title': row.get('title', row.get('name', file_path.stem)),
                            'content': content,
                            'source': f"dataset_{file_path.stem}",
                            'url': row.get('url', ''),
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

            chunks = self._split_text_into_chunks(content, file_path.stem)
            for i, chunk in enumerate(chunks):
                para = chunk['content'].strip()
                if len(para) > 80:
                    knowledge.append({
                        'title': chunk.get('title', f"{file_path.stem}_{i+1}"),
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

        print("   📖 爬取网络心理学词条...")
        crawled_results = self.crawler.crawl_all_sources(keywords)
        for item in crawled_results:
            knowledge.append({
                'title': item['title'],
                'content': item['content'],
                'source': item['source'],
                'url': item.get('url', ''),
                'issue_type': self._classify_content(item['content']),
                'type': 'crawled',
                'quality_score': item.get('quality_score', 0.5)
            })

        return knowledge

    def _extract_item_content(self, item: Dict) -> str:
        fields = [
            'content', 'text', 'article', 'body', 'description', 'summary',
            'answer', 'response', 'assistant', 'completion', 'output',
            'question', 'prompt', 'instruction'
        ]
        parts = []
        for field in fields:
            value = item.get(field)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
            elif isinstance(value, list):
                parts.extend(str(v).strip() for v in value if str(v).strip())
        return "\n".join(parts)

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

            {"title": "考试焦虑的短时稳定方法",
             "content": "考试焦虑出现时，可以先把目标从“立刻不焦虑”改成“让身体降一点速”。做三轮呼吸，吸气4秒、停2秒、呼气6秒；再写下最担心的结果，并补一句更平衡的解释，例如“这次考试很重要，但它不是我全部能力的证明”。最后只选择一个可执行动作：复习一道错题、整理一页公式或睡前停止刷题。",
             "source": "builtin", "issue_type": "academic anxiety", "type": "builtin"},

            {"title": "失恋后的反刍思维处理",
             "content": "失恋后反复想“如果当时我……”是常见反刍。处理方式不是强行忘记，而是给反刍限定时间：每天固定15分钟写下想法，时间结束后转向具体行为，如洗澡、散步、联系朋友。把问题从“为什么我不值得被爱”改写为“这段关系里我学到了什么、以后需要什么边界”。",
             "source": "builtin", "issue_type": "romantic breakup", "type": "builtin"},

            {"title": "职场压力的可控圈练习",
             "content": "面对职场压力时，可以把事情分为三圈：可控、可影响、不可控。可控包括今天先完成哪一步、如何沟通、何时休息；可影响包括和同事协调资源；不可控包括他人的评价和临时变化。行动优先放在可控圈，能减少无助感。",
             "source": "builtin", "issue_type": "workplace stress", "type": "builtin"},

            {"title": "人际冲突中的非暴力沟通",
             "content": "非暴力沟通包含观察、感受、需要、请求四步。比如不要说“你总是不尊重我”，可以说“刚才我说话时被打断了两次，我有些受伤，因为我需要被认真听见。下次能不能等我说完再回应？”这种表达降低攻击性，更容易让对方理解真实需求。",
             "source": "builtin", "issue_type": "interpersonal conflict", "type": "builtin"},

            {"title": "自我关怀的三句话",
             "content": "自我关怀不是纵容自己，而是在痛苦时用更有效的方式支持自己。可以练习三句话：第一，“我现在确实很难受”；第二，“人在这种处境下难受是正常的”；第三，“我可以先做一个小动作照顾自己”。这能降低羞耻和自责，帮助恢复行动力。",
             "source": "builtin", "issue_type": "mental health", "type": "builtin"},

            {"title": "睡前焦虑的关闭仪式",
             "content": "睡前焦虑常来自大脑仍在处理未完成事项。可以建立关闭仪式：写下明天要做的三件事，把担心写在纸上并注明“明天处理”；睡前30分钟远离屏幕；做渐进式肌肉放松，从脚趾到肩膀逐步紧绷再放松。重点是告诉身体今天已经结束。",
             "source": "builtin", "issue_type": "mental health", "type": "builtin"},

            {"title": "家庭关系中的边界表达",
             "content": "家庭边界不是冷漠，而是让关系可持续。可以用温和但清楚的句式：“我理解你担心我，但这个决定我想自己负责”；“我愿意听建议，但不接受贬低”；“这个话题我们都激动了，晚点再谈”。边界需要重复表达，不一定一次就被接受。",
             "source": "builtin", "issue_type": "family issues", "type": "builtin"},

            {"title": "情绪命名降低强度",
             "content": "把情绪准确命名可以降低情绪强度。与其只说“我很崩溃”，可以区分是委屈、害怕、羞耻、愤怒、失望还是孤独。命名后再问：这个情绪在保护我什么？它希望我注意什么？这样能从被情绪淹没，转向理解情绪的信息。",
             "source": "builtin", "issue_type": "mental health", "type": "builtin"},

            {"title": "低动力时期的行为激活",
             "content": "情绪低落时等待动力出现往往很难。行为激活建议先做极小行动，再让行动带来一点动力。可以选择2分钟任务：打开窗户、倒一杯水、洗脸、把桌面清出一个角落、走到楼下。完成后记录“我做到了”，强化可控感。",
             "source": "builtin", "issue_type": "mental health", "type": "builtin"},

            {"title": "危机风险识别与求助",
             "content": "如果出现持续的自伤念头、明确计划、无法保证自己安全，应该立即联系身边可信任的人，并寻求当地急救、危机热线或专业医疗帮助。AI助手只能提供支持性陪伴，不能替代危机干预。安全优先于隐私和面子。",
             "source": "builtin", "issue_type": "mental health", "type": "builtin"},
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
        """搜索相关知识。

        The vector index can return very weak matches when the knowledge base is
        small. We keep the threshold conservative so generic entries such as
        relaxation exercises do not appear for every query.
        """
        if not self.is_ready:
            return []
        candidates = self.index.search(
            query,
            k=max(k * 3, 6),
            issue_type=issue_type,
            min_score=0.18,
        )
        ranked = []
        for item in candidates:
            score = float(item.get("score", 0.0))
            if self._is_low_value_emotion_sample(item):
                continue
            if self._is_low_value_relaxation(item):
                continue
            adjusted_score = score
            item_issue = item.get("issue_type", "general")
            if issue_type and item_issue == issue_type:
                adjusted_score += 0.08
            if self._has_query_overlap(query, item):
                adjusted_score += 0.04
            if self._is_over_generic_relaxation(query, item):
                adjusted_score -= 0.12
            if adjusted_score >= 0.26:
                item = dict(item)
                item["score"] = adjusted_score
                item["raw_score"] = score
                ranked.append(item)

        if len(ranked) < k:
            seen = {item.get("id") for item in ranked}
            for item in self._keyword_fallback(query, issue_type):
                if item.get("id") in seen:
                    continue
                ranked.append(item)
                seen.add(item.get("id"))
                if len(ranked) >= k:
                    break

        ranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return ranked[:k]

    def _has_query_overlap(self, query: str, item: Dict) -> bool:
        query = (query or "").lower()
        text = f"{item.get('title', '')} {item.get('content', '')}".lower()
        keywords = [
            "开心", "高兴", "快乐", "难过", "伤心", "焦虑", "压力", "紧张", "睡眠",
            "考试", "学习", "工作", "分手", "失恋", "家庭", "父母", "朋友", "冲突",
            "happy", "sad", "anxiety", "stress", "exam", "study", "work", "breakup",
            "family", "conflict", "sleep", "relax",
        ]
        return any(word in query and word in text for word in keywords)

    def _is_over_generic_relaxation(self, query: str, item: Dict) -> bool:
        title = (item.get("title") or "").strip().lower()
        if "放松" not in title and "relax" not in title:
            return False
        query = (query or "").lower()
        relaxation_triggers = [
            "放松", "压力", "焦虑", "紧张", "失眠", "睡眠", "呼吸", "崩溃", "累",
            "stress", "anxiety", "tense", "sleep", "breath", "relax", "overwhelmed",
        ]
        return not any(word in query for word in relaxation_triggers)

    def _is_low_value_relaxation(self, item: Dict) -> bool:
        title = (item.get("title") or "").strip().lower()
        if title not in ("放松", "relax", "relaxation"):
            return False
        content = (item.get("content") or "").lower()
        useful_terms = ["呼吸", "肌肉", "焦虑", "压力", "正念", "冥想", "练习", "breath", "stress", "anxiety", "practice"]
        return not any(term in content for term in useful_terms)

    def _is_low_value_emotion_sample(self, item: Dict) -> bool:
        source = (item.get("source") or "").lower()
        title = (item.get("title") or "").lower()
        if "goemotions" in source:
            return True
        if title.startswith("情感表达_") or title.startswith("emotion_"):
            return True
        return False

    def _keyword_fallback(self, query: str, issue_type: str = None) -> List[Dict]:
        if not self.index:
            return []
        query = (query or "").lower()
        groups = [
            (["考试", "学习", "学业", "没考好", "exam", "study"], ["认知行为疗法", "CBT", "共情"]),
            (["焦虑", "紧张", "担心", "慌", "anxiety", "nervous", "worried"], ["焦虑", "正念", "冥想", "认知行为疗法", "CBT"]),
            (["压力", "累", "崩溃", "睡不着", "失眠", "sleep", "stress", "overwhelmed"], ["心理压力", "睡眠", "正念", "冥想", "焦虑"]),
            (["分手", "失恋", "前任", "breakup", "heartbreak"], ["分手", "情感", "情绪"]),
            (["朋友", "冲突", "吵架", "人际", "社交", "conflict", "friend"], ["人际关系", "社交", "共情"]),
            (["家庭", "父母", "家人", "family", "parent"], ["共情", "认知行为疗法", "心理健康"]),
        ]
        active_terms = []
        preferred_titles = []
        for triggers, titles in groups:
            if any(term in query for term in triggers):
                active_terms.extend(triggers)
                preferred_titles.extend(titles)
        if not preferred_titles:
            return []

        results = []
        for idx, item in enumerate(self.index.knowledge_base):
            if self._is_low_value_emotion_sample(item):
                continue
            if self._is_low_value_relaxation(item):
                continue
            item_issue = item.get("issue_type", "general")
            if issue_type and item_issue not in (issue_type, "general", "mental health"):
                continue
            title = item.get("title", "")
            content = item.get("content", "")
            haystack = f"{title} {content}".lower()
            score = 0.0
            if any(title_key.lower() in title.lower() or title_key.lower() in haystack for title_key in preferred_titles):
                score += 0.32
            score += min(0.12, 0.03 * sum(1 for term in active_terms if term in haystack))
            if score >= 0.30:
                results.append({
                    "id": item.get("id", idx),
                    "content": content,
                    "title": title or "未命名",
                    "source": item.get("source", "unknown"),
                    "score": score,
                    "raw_score": 0.0,
                    "type": item.get("type", "unknown"),
                    "issue_type": item_issue,
                })
        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return results

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

    def deduplicate_and_rebuild(self) -> Dict:
        """清理已有重复知识并重建 FAISS 索引。"""
        if not self.is_ready:
            self.load()
        if not self.index:
            return {'before': 0, 'after': 0, 'removed': 0}
        result = self.index.deduplicate_existing_items()
        if result.get('removed', 0) > 0:
            self.index.save(str(self.index_path))
            self.is_ready = True
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
