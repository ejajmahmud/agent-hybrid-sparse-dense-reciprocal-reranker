"""CLI for Hybrid Reciprocal Rank Reranker."""
import argparse
from .core import ReciprocalRankReranker

def main():
    parser = argparse.ArgumentParser(description="Hybrid Reciprocal Rank Reranker CLI")
    parser.add_argument("--demo", action="store_true", help="Demonstrate reciprocal rank fusion")
    args = parser.parse_args()

    reranker = ReciprocalRankReranker(k=60)
    dense_hits = ["docA", "docB", "docC"]
    sparse_hits = ["docB", "docD", "docA"]

    fused = reranker.fuse(dense_hits, sparse_hits, top_n=3)
    print("Fused Rankings:")
    for rank, (doc_id, score) in enumerate(fused, 1):
        print(f"  #{rank}: {doc_id} (RRF Score: {score:.5f})")

if __name__ == "__main__":
    main()
