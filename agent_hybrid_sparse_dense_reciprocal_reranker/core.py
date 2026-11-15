"""Core implementation of hybrid reciprocal rank fusion reranker."""
from typing import List, Dict, Tuple

class SearchResult:
    def __init__(self, doc_id: str, content: str):
        self.doc_id = doc_id
        self.content = content

class ReciprocalRankReranker:
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, dense_results: List[str], sparse_results: List[str], top_n: int = 5) -> List[Tuple[str, float]]:
        scores: Dict[str, float] = {}

        for rank, doc_id in enumerate(dense_results, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.k + rank)

        for rank, doc_id in enumerate(sparse_results, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.k + rank)

        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_docs[:top_n]
