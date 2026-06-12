"""
Robust psychology knowledge crawler.

The crawler keeps the original Baidu-Baike workflow, but adds:
- multiple URL candidates per concept;
- retrying HTTP session;
- generic article extraction for pages whose CSS classes changed;
- larger bilingual emotional-support keyword coverage;
- compatibility methods used by older knowledge-base code.
"""

from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Set
from urllib.parse import quote, urlencode

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class PsychologyCrawler:
    """Collect psychology and emotional recovery knowledge from public pages."""

    def __init__(self, rate_limit: float = 0.4, timeout: int = 18):
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
            "Connection": "keep-alive",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        retry = Retry(
            total=2,
            connect=2,
            read=2,
            backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.content_hashes: Set[str] = set()
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.network_blocked = False

        self.psychology_concepts = [
            "共情", "情绪调节", "情绪管理", "焦虑", "抑郁", "心理压力",
            "认知行为疗法", "认知重构", "自动化思维", "灾难化思维", "正念",
            "冥想", "放松训练", "腹式呼吸", "接地技术", "自我关怀",
            "心理韧性", "心理健康", "心理咨询", "社会支持", "人际关系",
            "沟通技巧", "亲密关系", "失恋", "分手", "家庭沟通", "亲子关系",
            "职场压力", "学业压力", "考试焦虑", "睡眠卫生", "压力管理",
            "emotional regulation", "cognitive behavioral therapy",
            "mindfulness", "self compassion", "grounding techniques",
            "stress management", "social support",
        ]
        self.curated_urls = {
            "认知行为疗法": [
                "https://zh.wikipedia.org/wiki/%E8%AE%A4%E7%9F%A5%E8%A1%8C%E4%B8%BA%E7%96%97%E6%B3%95",
                "https://en.wikipedia.org/wiki/Cognitive_behavioral_therapy",
            ],
            "正念冥想": [
                "https://zh.wikipedia.org/wiki/%E6%AD%A3%E5%BF%B5",
                "https://en.wikipedia.org/wiki/Mindfulness",
            ],
            "压力管理": [
                "https://en.wikipedia.org/wiki/Stress_management",
            ],
            "焦虑缓解": [
                "https://en.wikipedia.org/wiki/Anxiety",
            ],
            "情绪管理": [
                "https://en.wikipedia.org/wiki/Emotion_regulation",
            ],
            "共情技巧": [
                "https://zh.wikipedia.org/wiki/%E5%85%B1%E6%83%85",
                "https://en.wikipedia.org/wiki/Empathy",
            ],
            "自我关怀": [
                "https://en.wikipedia.org/wiki/Self-compassion",
            ],
        }

    def clean_text(self, text: str, max_length: int = 2600) -> str:
        if not text:
            return ""
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\[[0-9]+\]", "", text)
        text = re.sub(r"\s+", " ", text)
        text = text.replace("编辑", "").replace("播报", "")
        text = text.strip()
        if len(text) <= max_length:
            return text
        cut = max(text[:max_length].rfind("。"), text[:max_length].rfind("."), text[:max_length].rfind("\n"))
        if cut < max_length * 0.55:
            cut = max_length
        return text[: cut + 1].strip()

    def get_content_hash(self, content: str) -> str:
        normalized = re.sub(r"\s+", "", content[:500])
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def _request(self, url: str) -> Optional[str]:
        if self.network_blocked:
            return None
        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code != 200:
                return None
            if not response.encoding or response.encoding.lower() == "iso-8859-1":
                response.encoding = response.apparent_encoding or "utf-8"
            return response.text
        except Exception as exc:
            print(f"      request failed: {exc}")
            if "WinError 10013" in str(exc) or "访问权限不允许" in str(exc):
                self.network_blocked = True
                print("      network appears blocked; skip remaining web requests")
            return None

    def _extract_article_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "form"]):
            tag.decompose()

        selectors = [
            "div.para",
            "div.lemma-summary",
            "div.main-content",
            "main",
            "article",
            "div.content",
            "section",
            "p",
        ]
        parts: List[str] = []
        seen: Set[str] = set()
        for selector in selectors:
            for node in soup.select(selector):
                text = node.get_text(" ", strip=True)
                text = self.clean_text(text, max_length=1200)
                if len(text) < 45:
                    continue
                key = re.sub(r"\s+", "", text[:120])
                if key in seen:
                    continue
                seen.add(key)
                parts.append(text)
            if len(" ".join(parts)) > 500:
                break

        if not parts:
            body_text = soup.get_text(" ", strip=True)
            return self.clean_text(body_text)
        return self.clean_text("\n".join(parts))

    def _candidate_urls(self, concept: str) -> List[str]:
        encoded = quote(concept)
        urls = list(self.curated_urls.get(concept, []))
        urls.extend([
            f"https://baike.baidu.com/item/{encoded}?adapt=1",
            f"https://baike.baidu.com/item/{encoded}",
            f"https://zh.wikipedia.org/wiki/{encoded}",
        ])
        if re.search(r"[A-Za-z]", concept):
            title = concept.strip().replace(" ", "_")
            urls.extend([
                f"https://en.wikipedia.org/wiki/{quote(title)}",
                f"https://simple.wikipedia.org/wiki/{quote(title)}",
            ])
        return urls

    def crawl_concept(self, concept: str) -> Optional[Dict]:
        api_item = self._crawl_wikipedia_summary(concept)
        if api_item:
            return api_item
        for url in self._candidate_urls(concept):
            html = self._request(url)
            if not html:
                continue
            content = self._extract_article_text(html)
            if len(content) < 90:
                continue
            content_hash = self.get_content_hash(content)
            if content_hash in self.content_hashes:
                return None
            self.content_hashes.add(content_hash)
            return {
                "source": self._source_name(url),
                "title": concept,
                "content": content,
                "url": url,
                "type": "concept",
                "quality_score": self._quality_score(content, url),
                "issue_type": self.classify_issue_type(content + concept),
                "crawled_at": datetime.now().isoformat(),
            }
        return None

    def _crawl_wikipedia_summary(self, concept: str) -> Optional[Dict]:
        """Use Wikipedia REST summaries when normal HTML pages are blocked."""
        candidates = [concept]
        aliases = {
            "认知行为疗法": ["认知行为疗法", "Cognitive behavioral therapy"],
            "情绪管理": ["Emotion regulation"],
            "情绪调节": ["Emotion regulation"],
            "正念冥想": ["Mindfulness"],
            "压力管理": ["Stress management"],
            "自我关怀": ["Self-compassion"],
            "共情技巧": ["Empathy"],
            "焦虑缓解": ["Anxiety"],
        }
        candidates.extend(aliases.get(concept, []))
        endpoints = [
            "https://zh.wikipedia.org/api/rest_v1/page/summary/{}",
            "https://en.wikipedia.org/api/rest_v1/page/summary/{}",
        ]
        for title in dict.fromkeys(candidates):
            for endpoint in endpoints:
                if self.network_blocked:
                    return None
                url = endpoint.format(quote(title.replace(" ", "_")))
                try:
                    response = self.session.get(url, timeout=self.timeout)
                    if response.status_code != 200:
                        continue
                    data = response.json()
                    extract = self.clean_text(data.get("extract", ""), max_length=1800)
                    if len(extract) < 90:
                        continue
                    content_hash = self.get_content_hash(extract)
                    if content_hash in self.content_hashes:
                        return None
                    self.content_hashes.add(content_hash)
                    return {
                        "source": "wikipedia_summary",
                        "title": concept,
                        "content": extract,
                        "url": data.get("content_urls", {}).get("desktop", {}).get("page", url),
                        "type": "summary",
                        "quality_score": 0.7,
                        "issue_type": self.classify_issue_type(extract + concept),
                        "crawled_at": datetime.now().isoformat(),
                    }
                except Exception as exc:
                    if "WinError 10013" in str(exc) or "访问权限不允许" in str(exc):
                        self.network_blocked = True
                    continue
        return None

    def crawl_baike_mobile(self, concept: str) -> Optional[str]:
        """Backward-compatible API: return only article text."""
        item = self.crawl_concept(concept)
        return item["content"] if item else None

    def crawl_baike_psychology(self, keywords: List[str]) -> List[Dict]:
        """Backward-compatible API used by RAGKnowledgeBase."""
        return self.crawl_keywords(keywords)

    def crawl_zhihu_topic(self, pages: int = 1) -> List[Dict]:
        """Compatibility placeholder.

        Zhihu pages are frequently login/anti-bot protected, so the crawler avoids
        fragile scraping and returns an empty list instead of blocking the build.
        """
        return []

    def crawl_keywords(self, keywords: List[str]) -> List[Dict]:
        results = []
        for concept in keywords:
            if self.network_blocked:
                print("   network blocked, crawler stopped early")
                break
            print(f"   crawling: {concept}")
            item = self.crawl_concept(concept)
            if item:
                results.append(item)
                print(f"      ok ({len(item['content'])} chars, {item['source']})")
            else:
                print("      no usable content")
            time.sleep(self.rate_limit)
        return results

    def crawl_all_sources(self, keywords: Optional[List[str]] = None) -> List[Dict]:
        all_results = []
        concepts = keywords or self.psychology_concepts

        print("\n🕷️ 开始爬取心理学知识...")
        print("=" * 50)
        print(f"📚 关键词数量: {len(concepts)}")

        all_results.extend(self.crawl_keywords(concepts))

        print("\n" + "=" * 50)
        print("📊 爬取统计:")
        print(f"   总获取: {len(all_results)} 条")
        return all_results

    def _source_name(self, url: str) -> str:
        if "baike.baidu.com" in url:
            return "baidu_baike"
        if "wikipedia.org" in url:
            return "wikipedia"
        return "web"

    def _quality_score(self, content: str, url: str) -> float:
        score = 0.55
        if len(content) > 500:
            score += 0.15
        if len(content) > 1200:
            score += 0.1
        if "wikipedia.org" in url or "baike.baidu.com" in url:
            score += 0.1
        return min(score, 0.95)

    def classify_issue_type(self, text: str) -> str:
        text_lower = text.lower()
        categories = {
            "romantic breakup": ["分手", "失恋", "恋爱", "亲密关系", "breakup", "heartbreak", "relationship"],
            "interpersonal conflict": ["吵架", "冲突", "人际", "沟通", "朋友", "conflict", "communication"],
            "workplace stress": ["工作", "职场", "职业", "压力", "work", "career", "burnout"],
            "mental health": ["焦虑", "抑郁", "情绪", "心理", "失眠", "anxiety", "depression", "emotion"],
            "family issues": ["家庭", "父母", "家人", "亲子", "family", "parent"],
            "academic anxiety": ["学习", "考试", "学业", "成绩", "academic", "exam", "study"],
        }
        for issue_type, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return issue_type
        return "general"


def convert_crawled_to_knowledge(crawled_results: List[Dict]) -> List[Dict]:
    knowledge_items = []
    for item in crawled_results:
        knowledge_items.append({
            "title": item["title"],
            "content": item["content"],
            "source": item["source"],
            "url": item.get("url", ""),
            "issue_type": item.get("issue_type", "general"),
            "type": item.get("type", "crawled"),
            "quality_score": item.get("quality_score", 0.5),
            "crawled_at": item.get("crawled_at", ""),
        })
    return knowledge_items
