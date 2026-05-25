"""
公开数据集导入器 - 支持多种公开心理学数据集
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Optional
import jsonlines


class DatasetLoader:
    """公开数据集加载器"""

    def __init__(self, data_dir: Path = Path("./data/datasets")):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_psych8k(self, file_path: Path) -> List[Dict]:
        """
        加载 Psych8k 数据集（如果可用）

        Psych8k 是一个中文心理对话数据集
        格式: {"对话": "...", "标签": "..."}
        """
        knowledge_items = []

        if not file_path.exists():
            print(f"Psych8k 数据集未找到: {file_path}")
            return knowledge_items

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    content = data.get('对话', '')
                    if len(content) > 50:
                        knowledge_items.append({
                            'title': f"心理对话示例",
                            'content': content,
                            'source': 'psych8k',
                            'issue_type': self._classify_by_label(data.get('标签', '')),
                            'type': 'conversation'
                        })
            print(f"从 Psych8k 加载了 {len(knowledge_items)} 条知识")
        except Exception as e:
            print(f"加载 Psych8k 失败: {e}")

        return knowledge_items

    def load_psychology_articles(self, file_path: Path) -> List[Dict]:
        """
        加载心理学文章数据集

        格式: {"title": "...", "content": "...", "category": "..."}
        """
        knowledge_items = []

        if not file_path.exists():
            print(f"文章数据集未找到: {file_path}")
            return knowledge_items

        try:
            if file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        knowledge_items.append({
                            'title': item.get('title', '未命名'),
                            'content': item.get('content', ''),
                            'source': 'article_dataset',
                            'issue_type': self._classify_by_category(item.get('category', '')),
                            'type': 'article'
                        })
            elif file_path.suffix == '.jsonl':
                with jsonlines.open(file_path) as reader:
                    for item in reader:
                        knowledge_items.append({
                            'title': item.get('title', '未命名'),
                            'content': item.get('content', ''),
                            'source': 'article_dataset',
                            'issue_type': self._classify_by_category(item.get('category', '')),
                            'type': 'article'
                        })

            print(f"从文章数据集加载了 {len(knowledge_items)} 条知识")
        except Exception as e:
            print(f"加载文章数据集失败: {e}")

        return knowledge_items

    def load_csv_knowledge(self, file_path: Path,
                           title_col: str = 'title',
                           content_col: str = 'content',
                           category_col: str = 'category') -> List[Dict]:
        """从CSV文件加载知识"""
        knowledge_items = []

        if not file_path.exists():
            print(f"CSV文件未找到: {file_path}")
            return knowledge_items

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    content = row.get(content_col, '')
                    if len(content) > 50:
                        knowledge_items.append({
                            'title': row.get(title_col, '未命名'),
                            'content': content,
                            'source': 'csv_dataset',
                            'issue_type': self._classify_by_category(row.get(category_col, '')),
                            'type': 'csv'
                        })
            print(f"从CSV加载了 {len(knowledge_items)} 条知识")
        except Exception as e:
            print(f"加载CSV失败: {e}")

        return knowledge_items

    def load_sample_psychology_data(self) -> List[Dict]:
        """加载样例心理学数据（内置）"""
        sample_data = [
            {
                "title": "情绪调节策略 - 认知重评",
                "content": """认知重评是一种有效的情绪调节策略。它指的是改变对情绪事件的理解和解释方式。
                具体步骤：
                1. 识别当前情绪和触发事件
                2. 问自己："还有其他的解释方式吗？"
                3. 尝试从不同角度看待事情
                4. 考虑长期视角："一周/一月/一年后这件事还重要吗？"
                研究表明，经常练习认知重评的人焦虑和抑郁水平更低，生活满意度更高。""",
                "category": "情绪调节"
            },
            {
                "title": "正念减压疗法",
                "content": """正念减压疗法(MBSR)由Jon Kabat-Zinn博士开发，被证明对焦虑、抑郁、慢性疼痛有效。
                核心练习：
                - 正念呼吸：专注于呼吸的每个瞬间
                - 身体扫描：从脚趾到头顶逐一感受身体部位
                - 正念行走：专注于行走时身体的每个动作
                - 日常正念：吃饭、洗碗时的正念练习
                每天练习10-15分钟，坚持8周会有显著效果。""",
                "category": "正念冥想"
            },
            {
                "title": "有效沟通的DEAR MAN技巧",
                "content": """DEAR MAN是DBT疗法中的人际效能技巧：
                D (Describe): 描述事实情况
                E (Express): 表达感受和观点
                A (Assert): 清晰地表达请求或拒绝
                R (Reinforce): 提前说明请求被满足的好处
                M (Mindful): 保持专注，不偏离话题
                A (Appear confident): 展现自信
                N (Negotiate): 愿意协商找到双方都能接受的方案
                这个技巧特别适合解决人际关系中的冲突。""",
                "category": "人际关系"
            },
            {
                "title": "分手后的心理恢复阶段",
                "content": """心理学研究表明，分手后的恢复通常会经历以下阶段：
                1. 否认期："这不是真的，我们还会复合"
                2. 愤怒期："都是TA的错"或"都是我的错"
                3. 协商期："如果当初...就好了"
                4. 抑郁期：感到悲伤、空虚、失去动力
                5. 接受期：开始接受现实，重新规划生活
                
                建议：
                - 允许自己经历所有情绪，不要压抑
                - 保持日常规律，不要完全改变生活节奏
                - 寻求支持系统（朋友、家人、专业帮助）
                - 给自己时间，没有"应该"多久恢复的标准""",
                "category": "分手恢复"
            },
            {
                "title": "工作压力管理策略",
                "content": """职场压力管理的实用策略：
                1. 设置工作边界
                   - 明确工作时间，避免无休止加班
                   - 学会说"不"，优先处理重要任务
                
                2. 任务分解法
                   - 将大任务分解为小步骤
                   - 每完成一步就给自己肯定
                
                3. 定期休息
                   - 使用番茄工作法(25分钟工作+5分钟休息)
                   - 每小时站起来走动几分钟
                
                4. 建立支持系统
                   - 与信任的同事交流
                   - 寻找职场导师
                
                5. 区分可控与不可控
                   - 专注于自己能改变的事情
                   - 接受无法控制的部分，减少内耗""",
                "category": "工作压力"
            },
            {
                "title": "CBT认知扭曲识别指南",
                "content": """认知行为疗法中常见的认知扭曲类型：
                
                1. 非黑即白思维
                例子："如果这次面试没通过，我就是个彻底的失败者"
                重构："一次面试结果不能定义我的全部价值"
                
                2. 灾难化
                例子："如果我在演讲中说错话，所有人都会嘲笑我"
                重构："即使说错话，大多数人也不会太在意"
                
                3. 过度概括
                例子:"上次约会搞砸了，我这辈子都不会有好的感情了"
                重构:"一次失败的经历不代表永远失败"
                
                4. 读心术
                例子："TA没回消息，肯定是讨厌我了"
                重构："有很多原因可能导致TA没及时回复"
                
                5. 情感推理
                例子："我感觉自己很没用，所以我一定很没用"
                重构:"感受不等于事实，可以质疑这种感觉"
                
                练习：记录每天的自动负面思维，尝试找出其中的认知扭曲类型""",
                "category": "认知行为疗法"
            }
        ]

        knowledge_items = []
        for item in sample_data:
            knowledge_items.append({
                'title': item['title'],
                'content': item['content'],
                'source': 'sample_dataset',
                'issue_type': self._classify_by_category(item['category']),
                'type': 'sample'
            })

        print(f"加载了 {len(knowledge_items)} 条样例心理学知识")
        return knowledge_items

    def _classify_by_label(self, label: str) -> str:
        """根据标签分类"""
        label_map = {
            '焦虑': 'mental health',
            '抑郁': 'mental health',
            '分手': 'romantic breakup',
            '失恋': 'romantic breakup',
            '工作': 'workplace stress',
            '职场': 'workplace stress',
            '家庭': 'family issues',
            '人际': 'interpersonal conflict',
        }

        for key, value in label_map.items():
            if key in label:
                return value
        return 'general'

    def _classify_by_category(self, category: str) -> str:
        """根据类别分类"""
        category_map = {
            '情绪调节': 'mental health',
            '正念冥想': 'mental health',
            '人际关系': 'interpersonal conflict',
            '分手恢复': 'romantic breakup',
            '工作压力': 'workplace stress',
            '认知行为疗法': 'mental health',
            'anxiety': 'mental health',
            'depression': 'mental health',
            'breakup': 'romantic breakup',
            'work stress': 'workplace stress'
        }

        for key, value in category_map.items():
            if key in category:
                return value
        return 'general'

    def load_all_datasets(self, dataset_files: Dict[str, Path]) -> List[Dict]:
        """
        加载所有数据集

        Args:
            dataset_files: 数据集文件字典 {'psych8k': path, 'articles': path, 'csv': path}

        Returns:
            合并后的知识列表
        """
        all_knowledge = []

        # 1. 加载样例数据（总是可用）
        all_knowledge.extend(self.load_sample_psychology_data())

        # 2. 加载 Psych8k（如果存在）
        if 'psych8k' in dataset_files:
            all_knowledge.extend(self.load_psych8k(dataset_files['psych8k']))

        # 3. 加载文章数据集
        if 'articles' in dataset_files:
            all_knowledge.extend(self.load_psychology_articles(dataset_files['articles']))

        # 4. 加载 CSV 数据集
        if 'csv' in dataset_files:
            all_knowledge.extend(self.load_csv_knowledge(dataset_files['csv']))

        return all_knowledge