"""
专业心理学知识爬虫 - 使用移动版百度百科
"""

import requests
from bs4 import BeautifulSoup
import time
import re
from typing import List, Dict, Set
from datetime import datetime
import hashlib


class PsychologyCrawler:
    """专业心理学知识爬虫 - 移动版百度百科"""

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        self.content_hashes: Set[str] = set()
        self.rate_limit = 0.5

        # 确保可以成功获取的心理学词条
        self.psychology_concepts = [
            "共情", "情绪", "焦虑", "抑郁", "心理压力",
            "人际关系", "自我认知", "情感", "心理健康", "心理咨询",
            "正念", "冥想", "放松", "睡眠", "社交"
        ]

    def clean_text(self, text: str) -> str:
        """清洗文本"""
        if not text:
            return ""
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        # 限制长度
        if len(text) > 1500:
            cut_point = text[:1500].rfind('。')
            if cut_point == -1:
                cut_point = 1500
            text = text[:cut_point + 1]
        return text.strip()

    def get_content_hash(self, content: str) -> str:
        """计算内容哈希"""
        normalized = content[:300].strip().replace(' ', '').replace('\n', '')
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()

    def crawl_baike_mobile(self, concept: str) -> str:
        """使用移动版百度百科爬取概念"""
        try:
            # 使用移动版URL
            url = f"https://baike.baidu.com/item/{concept}?adapt=1"

            response = requests.get(url, headers=self.headers, timeout=15)
            response.encoding = 'utf-8'

            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, 'html.parser')

            content_parts = []

            # 移动版的内容在 div class="para" 中
            for para in soup.find_all('div', class_='para'):
                text = para.get_text(strip=True)
                if len(text) > 30:
                    content_parts.append(text)

            # 如果没有找到，尝试其他选择器
            if not content_parts:
                for para in soup.find_all('p'):
                    text = para.get_text(strip=True)
                    if len(text) > 50:
                        content_parts.append(text)

            if not content_parts:
                return None

            return '\n'.join(content_parts)

        except Exception as e:
            print(f"      ❌ 请求失败: {e}")
            return None

    def crawl_all_sources(self) -> List[Dict]:
        """从所有来源爬取数据"""
        all_results = []

        print("\n🕷️ 开始爬取心理学知识...")
        print("=" * 50)

        print("\n📚 爬取百度百科心理学概念...")

        for concept in self.psychology_concepts:
            try:
                print(f"   正在爬取: {concept}")

                content = self.crawl_baike_mobile(concept)

                if not content:
                    print(f"      ⚠️ 未找到内容")
                    continue

                clean_content = self.clean_text(content)

                if not clean_content or len(clean_content) < 80:
                    print(f"      ⚠️ 内容过短 ({len(clean_content) if clean_content else 0}字)")
                    continue

                # 内容哈希去重
                content_hash = self.get_content_hash(clean_content)
                if content_hash in self.content_hashes:
                    print(f"      ⚠️ 内容重复")
                    continue
                self.content_hashes.add(content_hash)

                # 判断问题类型
                issue_type = self.classify_issue_type(clean_content + concept)

                all_results.append({
                    'source': 'baike',
                    'title': concept,
                    'content': clean_content,
                    'url': f"https://baike.baidu.com/item/{concept}",
                    'type': 'concept',
                    'quality_score': 0.7,
                    'issue_type': issue_type,
                    'crawled_at': datetime.now().isoformat()
                })

                print(f"      ✅ 成功 ({len(clean_content)}字)")
                time.sleep(self.rate_limit)

            except Exception as e:
                print(f"      ❌ 失败: {e}")

        print(f"\n📊 百度百科: 获取 {len(all_results)} 条")

        print("\n" + "=" * 50)
        print(f"📊 爬取统计:")
        print(f"   总获取: {len(all_results)} 条")

        return all_results

    def classify_issue_type(self, text: str) -> str:
        """分类问题类型"""
        text_lower = text.lower()

        if any(k in text_lower for k in ['分手', '失恋', '恋爱', '感情']):
            return 'romantic breakup'
        elif any(k in text_lower for k in ['吵架', '冲突', '人际', '沟通', '朋友']):
            return 'interpersonal conflict'
        elif any(k in text_lower for k in ['工作', '职场', '职业', '压力']):
            return 'workplace stress'
        elif any(k in text_lower for k in ['焦虑', '抑郁', '情绪', '心理', '失眠']):
            return 'mental health'
        elif any(k in text_lower for k in ['家庭', '父母', '家人', '亲子']):
            return 'family issues'
        elif any(k in text_lower for k in ['学习', '考试', '学业', '成绩']):
            return 'academic anxiety'
        else:
            return 'general'


def convert_crawled_to_knowledge(crawled_results: List[Dict]) -> List[Dict]:
    """将爬虫结果转换为知识库格式"""
    knowledge_items = []

    for item in crawled_results:
        knowledge_items.append({
            'title': item['title'],
            'content': item['content'],
            'source': item['source'],
            'issue_type': item.get('issue_type', 'general'),
            'type': item['type'],
            'quality_score': item.get('quality_score', 0.5),
            'crawled_at': item.get('crawled_at', '')
        })

    return knowledge_items