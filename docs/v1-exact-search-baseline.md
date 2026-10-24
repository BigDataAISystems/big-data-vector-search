V1 â€” Exact Vector Search Baseline



Dataset:

&#x20;   MS MARCO / BEIR



Embedding model:

&#x20;   sentence-transformers/all-MiniLM-L6-v2



Dimension:

&#x20;   384



Normalization:

&#x20;   L2



Index:

&#x20;   FAISS IndexFlatIP



Similarity:

&#x20;   Inner product / cosine for normalized vectors



Scale points:

&#x20;   1K

&#x20;   10K

&#x20;   25K

&#x20;   50K

&#x20;   100K



Evaluation queries:

&#x20;   Fixed across every scale point



Primary measurements:

&#x20;   corpus load time

&#x20;   document embedding time

&#x20;   embedding throughput

&#x20;   index build time

&#x20;   index size

&#x20;   memory

&#x20;   mean query latency

&#x20;   median query latency

&#x20;   P95 query latency

&#x20;   relevance Recall@K



Future V2 output:

&#x20;   exact Top-K neighbor IDs retained as ANN ground truth
