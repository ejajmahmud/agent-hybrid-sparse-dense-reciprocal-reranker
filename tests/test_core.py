"""Automated test suite for Hybrid Reciprocal Rank Reranker."""
import unittest
from agent_hybrid_sparse_dense_reciprocal_reranker.core import ReciprocalRankReranker

class TestReciprocalRankReranker(unittest.TestCase):
    def test_fusion_prioritizes_common_hits(self):
        reranker = ReciprocalRankReranker(k=60)
        # docB appears in both lists near the top
        dense = ["docA", "docB", "docC"]
        sparse = ["docB", "docX", "docY"]
        fused = reranker.fuse(dense, sparse, top_n=3)

        top_doc = fused[0][0]
        self.assertEqual(top_doc, "docB")
