"""
FAISS向量索引封装 - 支持增量累加和自动去重
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Set
import os
import hashlib
import sys

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

    def _load_model(self):
        """加载嵌入模型（使用本地缓存）"""
        if self.embedding_model is None:
            print("📦 正在加载嵌入模型（首次需要下载约471MB，请耐心等待）...")
            print("   如果下载慢，请确保网络正常或使用代理")

            # 确保缓存目录存在
            cache_dir = Path("./data/cache/sentence-transformers")
            cache_dir.mkdir(parents=True, exist_ok=True)

            try:
                # 延迟导入 sentence_transformers
                from sentence_transformers import SentenceTransformer

                # 设置超时和重试
                self.embedding_model = SentenceTransformer(
                    'paraphrase-multilingual-MiniLM-L12-v2',
                    cache_folder=str(cache_dir)
                )
                print("✅ 嵌入模型加载完成")
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                print("   请检查网络连接")
                print("   或者手动下载模型: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
                raise
        return self.embedding_model

    def _get_content_hash(self, content: str) -> str:
        """计算内容哈希用于去重"""
        normalized = content[:300].strip().replace(' ', '').replace('\n', '').replace('\r', '')
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()

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
        for item in self.knowledge_base:
            h = self._get_content_hash(item['content'])
            self.content_hashes.add(h)
            source_key = self._get_source_key(item)
            self.source_records.add(source_key)

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
        self.add_incremental(knowledge_base, show_progress)

    def add_incremental(self, new_items: List[Dict], show_progress: bool = True) -> Dict:
        """增量添加新知识（自动去重）"""
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

            if content_hash in self.content_hashes:
                duplicate_count += 1
                continue

            if source_key in self.source_records:
                duplicate_count += 1
                continue

            unique_new_items.append(item)
            self.content_hashes.add(content_hash)
            self.source_records.add(source_key)

        if not unique_new_items:
            print(f"📭 没有发现新知识（{duplicate_count} 条重复）")
            return {'total': len(new_items), 'added': 0, 'duplicates': duplicate_count,
                    'total_after': len(self.knowledge_base)}

        print(f"✨ 发现 {len(unique_new_items)} 条新知识（{duplicate_count} 条重复）")

        texts = [item['content'] for item in unique_new_items]
        embeddings = model.encode(texts, show_progress_bar=show_progress)

        if self.index is None:
            if self.use_ivf and len(unique_new_items) > 100:
                nlist = min(100, max(1, len(unique_new_items) // 10))
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
                self.index.train(embeddings.astype('float32'))
            else:
                self.index = faiss.IndexFlatL2(self.dimension)

        self.index.add(embeddings.astype('float32'))

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
        query_emb = model.encode([query])

        k_search = min(k * 2, self.index.ntotal)
        distances, indices = self.index.search(query_emb.astype('float32'), k_search)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1 or idx >= len(self.knowledge_base):
                continue

            score = float(1 / (1 + dist))
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
                'use_ivf': self.use_ivf
            }, f)

        print(f"💾 索引已保存: {path}.faiss ({len(self.knowledge_base)} 条)")

    def get_stats(self) -> Dict:
        """获取索引统计信息"""
        return {
            'total_entries': len(self.knowledge_base),
            'index_type': 'IVF' if self.use_ivf else 'FlatL2',
            'dimension': self.dimension,
            'is_trained': self.index.is_trained if self.index else False
        }

    def get_all_knowledge(self) -> List[Dict]:
        """获取所有知识"""
        return self.knowledge_base.copy()


# 向后兼容别名
VectorIndex = FAISSVectorIndex