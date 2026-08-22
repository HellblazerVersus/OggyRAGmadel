import time
from src.retrieval.bm25_retriever import BM25Retriever

def main():
    print("Initializing BM25 Retriever...")
    t0 = time.time()
    retriever = BM25Retriever(metadata_path="data/processed/passage_metadata.json", top_k=5)
    print(f"Initialization took {time.time() - t0:.4f} seconds")
    
    query = "How to test latency?"
    print(f"\nQuerying: '{query}'")
    
    t0 = time.time()
    result, embed_ms, search_ms = retriever.retrieve(query)
    total_ms = (time.time() - t0) * 1000
    
    print(f"Tokenization (Embed) Time: {embed_ms:.2f} ms")
    print(f"BM25 Search Time: {search_ms:.2f} ms")
    print(f"Total Retrieval Time: {total_ms:.2f} ms")
    print(f"\nTop Score: {result.top_score:.4f}")
    if result.passages:
        print(f"Top Result text: {result.passages[0].text[:100]}...")

if __name__ == "__main__":
    main()
