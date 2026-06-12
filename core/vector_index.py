"""
FAISS向量索引封装 - 支持增量累加、自动去重和知识分块
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Set
import os
import hashlib
import sys
import re

# 设置HuggingFace镜像和缓存目录
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = './data/cache'
os.environ['TRANSFORMERS_CACHE'] = './data/cache/transformers'


class FAISSVectorIndex:
    """FAISS向量索引类 - 支持增量累加和自动去重"""

    def __init__(self, dimension: int = 384, use_ivf: bool = False):
        self.dimension = dimension
        self.use_ivf = use_ivf
        self.index = None
        self.knowledge_base: List[Dict] = []
        self.embedding_model = None
        self.content_hashes: Set[str] = set()
        self.source_records: Set[str] = set()
        self.title_content_records: Set[str] = set()
        self.metric = "ip"

    def _load_model(self):
        """加载嵌入模型（使用本地缓存）"""
        if self.embedding_model is None:
            print("📦 正在加载本地嵌入模型...")

            # 确保缓存目录存在
            cache_dir = Path("./data/cache/sentence-transformers")
            cache_dir.mkdir(parents=True, exist_ok=True)
            allow_download = os.environ.get("ALLOW_EMBEDDING_MODEL_DOWNLOAD", "0") == "1"

            try:
                # 延迟导入 sentence_transformers
                from sentence_transformers import SentenceTransformer

                # Desktop inference should be low-latency: use local cache by
                # default and avoid long network retries during a chat turn.
                self.embedding_model = SentenceTransformer(
                    'paraphrase-multilingual-MiniLM-L12-v2',
                    cache_folder=str(cache_dir),
                    local_files_only=not allow_download,
                )
                print("✅ 嵌入模型加载完成")
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                print("   请手动放入多语言句向量模型到 data/cache 目录")
                print("   或设置 ALLOW_EMBEDDING_MODEL_DOWNLOAD=1 后运行知识库构建脚本下载")
                raise
        return self.embedding_model

    def _normalize_for_hash(self, text: str) -> str:
        text = re.sub(r'\s+', '', text or '')
        text = re.sub(r'[，。！？、；：“”‘’"\'.,!?;:()\[\]{}<>《》【】\-—_]+', '', text)
        return text.lower()

    def _get_content_hash(self, content: str) -> str:
        """计算内容哈希用于去重"""
        normalized = self._normalize_for_hash(content)[:800]
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()

    def _get_title_content_key(self, item: Dict) -> str:
        title = self._normalize_for_hash(item.get('title', ''))[:80]
        content = self._normalize_for_hash(item.get('content', ''))[:240]
        return hashlib.md5(f"{title}:{content}".encode('utf-8')).hexdigest()

    def _get_source_key(self, item: Dict) -> str:
        """生成来源唯一标识"""
        if item.get('url'):
            return f"url:{item['url']}"
        else:
            title = item.get('title', '')[:50]
            source = item.get('source', 'unknown')
            content_preview = item.get('content', '')[:100]
            return f"source:{source}:{title}:{content_preview}"

    def _update_hash_set(self):
        """从现有知识库更新哈希集合"""
        self.content_hashes.clear()
        self.source_records.clear()
        self.title_content_records.clear()
        for item in self.knowledge_base:
            h = self._get_content_hash(item['content'])
            self.content_hashes.add(h)
            source_key = self._get_source_key(item)
            self.source_records.add(source_key)
            self.title_content_records.add(self._get_title_content_key(item))

    def load(self, path: str) -> bool:
        """从磁盘加载已有索引（不需要加载模型）"""
        path = Path(path)
        faiss_path = path.with_suffix('.faiss')
        pkl_path = path.with_suffix('.pkl')

        if not faiss_path.exists() or not pkl_path.exists():
            print(f"⚠️ 索引文件不存在: {faiss_path}")
            return False

        try:
            self.index = faiss.read_index(str(faiss_path))
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                self.knowledge_base = data['knowledge_base']
                self.dimension = data.get('dimension', 384)
                self.use_ivf = data.get('use_ivf', False)
                self.metric = data.get('metric', 'l2')

            self._update_hash_set()

            print(f"✅ 索引加载成功！共 {self.index.ntotal} 条知识")
            return True
        except Exception as e:
            print(f"❌ 加载索引失败: {e}")
            return False

    def build_index(self, knowledge_base: List[Dict], show_progress: bool = True):
        """构建新索引（覆盖模式）"""
        self.knowledge_base = []
        self.content_hashes.clear()
        self.source_records.clear()
        self.title_content_records.clear()
        self.index = None
        self.metric = "ip"
        self.add_incremental(knowledge_base, show_progress)

    def deduplicate_existing_items(self) -> Dict:
        """Remove duplicate records already stored in memory."""
        unique_items = []
        seen_content = set()
        seen_source = set()
        seen_title_content = set()
        duplicate_count = 0

        for item in self.knowledge_base:
            content = item.get('content', '')
            if not content:
                duplicate_count += 1
                continue
            content_hash = self._get_content_hash(content)
            source_key = self._get_source_key(item)
            title_content_key = self._get_title_content_key(item)
            if (
                content_hash in seen_content
                or source_key in seen_source
                or title_content_key in seen_title_content
            ):
                duplicate_count += 1
                continue
            clean_item = dict(item)
            clean_item.pop('id', None)
            unique_items.append(clean_item)
            seen_content.add(content_hash)
            seen_source.add(source_key)
            seen_title_content.add(title_content_key)

        before = len(self.knowledge_base)
        if duplicate_count:
            self.build_index(unique_items, show_progress=True)
        return {
            'before': before,
            'after': len(unique_items),
            'removed': duplicate_count,
        }

    def add_incremental(self, new_items: List[Dict], show_progress: bool = True) -> Dict:
        """增量添加新知识（自动去重）"""
        new_items = self._prepare_items(new_items)
        if not new_items:
            return {'total': 0, 'added': 0, 'duplicates': 0, 'total_after': len(self.knowledge_base)}

        # 只有在需要添加新知识时才加载模型
        model = self._load_model()

        unique_new_items = []
        duplicate_count = 0

        for item in new_items:
            content = item.get('content', '')
            if not content:
                continue

            content_hash = self._get_content_hash(content)
            source_key = self._get_source_key(item)
            title_content_key = self._get_title_content_key(item)

            if content_hash in self.content_hashes:
                duplicate_count += 1
                continue

            if source_key in self.source_records:
                duplicate_count += 1
                continue

            if title_content_key in self.title_content_records:
                duplicate_count += 1
                continue

            unique_new_items.append(item)
            self.content_hashes.add(content_hash)
            self.source_records.add(source_key)
            self.title_content_records.add(title_content_key)

        if not unique_new_items:
            print(f"📭 没有发现新知识（{duplicate_count} 条重复）")
            return {'total': len(new_items), 'added': 0, 'duplicates': duplicate_count,
                    'total_after': len(self.knowledge_base)}

        print(f"✨ 发现 {len(unique_new_items)} 条新知识（{duplicate_count} 条重复）")

        texts = [self._embedding_text(item) for item in unique_new_items]
        embeddings = model.encode(
            texts,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            batch_size=32,
        ).astype('float32')

        if self.index is None:
            if self.use_ivf and len(unique_new_items) > 100:
                nlist = min(100, max(1, len(unique_new_items) // 10))
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                self.index.train(embeddings)
            else:
                self.index = faiss.IndexFlatIP(self.dimension)
            self.metric = "ip"

        self.index.add(embeddings)

        start_id = len(self.knowledge_base)
        for i, item in enumerate(unique_new_items):
            item['id'] = start_id + i
            self.knowledge_base.append(item)

        result = {
            'total': len(new_items),
            'added': len(unique_new_items),
            'duplicates': duplicate_count,
            'total_after': len(self.knowledge_base)
        }

        print(f"   ✅ 已添加 {len(unique_new_items)} 条新知识")
        print(f"   📊 知识库现有: {len(self.knowledge_base)} 条")

        return result

    def search(self, query: str, k: int = 5, issue_type: Optional[str] = None,
               min_score: float = 0.2) -> List[Dict]:
        """搜索相似知识（只在搜索时加载模型）"""
        if self.index is None or self.index.ntotal == 0:
            return []

        # 只有在搜索时才加载模型
        model = self._load_model()
        query_emb = model.encode([query], normalize_embeddings=True).astype('float32')

        k_search = min(max(k * 4, k), self.index.ntotal)
        distances, indices = self.index.search(query_emb, k_search)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1 or idx >= len(self.knowledge_base):
                continue

            score = float(dist) if self.metric == "ip" else float(1 / (1 + dist))
            if score < min_score:
                continue

            item = self.knowledge_base[idx]

            if issue_type and item.get('issue_type'):
                if item['issue_type'] != issue_type and item['issue_type'] != 'general':
                    continue

            results.append({
                'id': item.get('id', idx),
                'content': item['content'],
                'title': item.get('title', '未命名'),
                'source': item.get('source', 'unknown'),
                'score': score,
                'type': item.get('type', 'unknown'),
                'issue_type': item.get('issue_type', 'general')
            })

        return results[:k]

    def _prepare_items(self, items: List[Dict]) -> List[Dict]:
        prepared = []
        for item in items:
            content = self._clean_content(item.get('content', ''))
            if len(content) < 60:
                continue
            chunks = self._chunk_content(content)
            for idx, chunk in enumerate(chunks):
                chunk_item = dict(item)
                chunk_item['content'] = chunk
                if len(chunks) > 1:
                    chunk_item['title'] = f"{item.get('title', 'untitled')}_part{idx + 1}"
                    chunk_item['parent_title'] = item.get('title', '')
                    chunk_item['chunk_index'] = idx + 1
                    chunk_item['chunk_total'] = len(chunks)
                prepared.append(chunk_item)
        return prepared

    def _clean_content(self, text: str) -> str:
        import re

        text = re.sub(r'\s+', ' ', text or '').strip()
        text = re.sub(r'(相关推荐|参考资料|免责声明|本文编辑|返回顶部).*$', '', text)
        return text.strip()

    def _chunk_content(self, text: str, max_chars: int = 900, overlap: int = 120) -> List[str]:
        if len(text) <= max_chars:
            return [text]
        sentences = []
        for part in text.replace('\n', ' ').split('。'):
            part = part.strip()
            if part:
                sentences.append(part + '。')
        if len(sentences) <= 1:
            return [text[i:i + max_chars] for i in range(0, len(text), max_chars - overlap)]

        chunks = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) > max_chars and current:
                chunks.append(current.strip())
                current = current[-overlap:] + sentence
            else:
                current += sentence
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def _embedding_text(self, item: Dict) -> str:
        title = item.get('title', '')
        issue_type = item.get('issue_type', '')
        return f"{title}\n{issue_type}\n{item['content']}"

    def save(self, path: str):
        """保存索引到磁盘"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, str(path.with_suffix('.faiss')))

        with open(path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump({
                'knowledge_base': self.knowledge_base,
                'dimension': self.dimension,
                'use_ivf': self.use_ivf,
                'metric': self.metric
            }, f)

        print(f"💾 索引已保存: {path}.faiss ({len(self.knowledge_base)} 条)")

    def get_stats(self) -> Dict:
        """获取索引统计信息"""
        return {
            'total_entries': len(self.knowledge_base),
            'index_type': ('IVF' if self.use_ivf else 'Flat') + f"_{self.metric.upper()}",
            'dimension': self.dimension,
            'is_trained': self.index.is_trained if self.index else False
        }

    def get_all_knowledge(self) -> List[Dict]:
        """获取所有知识"""
        return self.knowledge_base.copy()


# 向后兼容别名
VectorIndex = FAISSVectorIndex
