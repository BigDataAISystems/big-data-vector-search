\# V1 â€” Exact Vector Search Baseline



\## Big Data Vector Search



V1 establishes the controlled \*\*exact vector-search baseline\*\* for the Big Data Vector Search project.



The objective of V1 is to characterize how exact semantic vector search behaves as the indexed collection grows while holding the evaluation workload, embedding model, vector dimensionality, search algorithm, and retrieval parameters constant.



V1 also produces validated exact nearest-neighbor results that will serve as the algorithmic ground truth for future Approximate Nearest Neighbor (ANN) experiments.



\---



\## 1. Research Objective



The primary V1 research question is:



> \*\*How does exact semantic vector search scale as the number of indexed document vectors increases?\*\*



The experiments examine the effects of corpus size on:



\- document embedding time;

\- embedding throughput;

\- exact index construction time;

\- index storage;

\- query-search latency;

\- process memory;

\- relevance-based retrieval quality.



V1 deliberately uses an exact vector index so that no approximation error is introduced.



This creates a reference point for future ANN experiments.



\---



\## 2. Experimental Architecture



The V1 pipeline is:



```text

MS MARCO Corpus

&#x20;     â†“

Controlled Nested Benchmark

&#x20;     â†“

Sentence Transformer

&#x20;     â†“

Normalized 384-D Embeddings

&#x20;     â†“

FAISS IndexFlatIP

&#x20;     â†“

300 Fixed Query Embeddings

&#x20;     â†“

Exact Top-K Search

&#x20;     â†“

Latency + Memory + Recall

&#x20;     â†“

Validated Exact Neighbor Ground Truth

```



\---



\## 3. Dataset



The benchmark is based on the \*\*MS MARCO\*\* collection distributed through the BEIR dataset ecosystem.



The source corpus contains approximately:



```text

8.84 million documents/passages

```



V1 constructs controlled nested subsets:



```text

1K

10K

25K

50K

100K

```



with the nesting property:



\\\[

S\_{1K}

\\subset

S\_{10K}

\\subset

S\_{25K}

\\subset

S\_{50K}

\\subset

S\_{100K}

\\]



This allows corpus size to change while preserving the evaluation workload.



\---



\## 4. Evaluation Queries



The experiments use:



```text

300 fixed evaluation queries

```



selected deterministically from the MS MARCO validation relevance judgments.



The random seed is:



```text

42

```



The validation qrels contain 6,980 eligible queries.



A deterministic sample of 300 queries is selected and reused at every corpus size.



\---



\## 5. Relevant Documents



The 300 selected queries require:



```text

314 positive query-document relevance pairs

314 unique relevant documents

```



Every benchmark subset contains all 314 required relevant documents.



The smallest 1K corpus therefore contains:



```text

314 required relevant documents

686 deterministic distractor documents

```



As corpus size increases, additional distractor documents are introduced while the query set and required relevant-document set remain fixed.



This design allows the effect of an increasingly large retrieval search space to be studied directly.



\---



\## 6. Benchmark Construction



The benchmark preparation program is:



```text

src/prepare\_msmarco\_benchmark.py

```



The program:



1\. loads MS MARCO queries;

2\. loads validation relevance judgments;

3\. deterministically selects 300 eligible queries;

4\. identifies all positively relevant documents;

5\. scans the MS MARCO corpus;

6\. retains every required relevant document;

7\. selects distractor documents using deterministic reservoir sampling;

8\. creates one master corpus;

9\. derives nested corpus subsets;

10\. writes benchmark provenance and metadata.



The generated benchmark is independently validated by:



```text

src/validate\_v1\_benchmark.py

```



Validation verifies:



\- exactly 300 unique queries;

\- no empty queries;

\- positive relevance judgments;

\- expected corpus cardinality;

\- unique document IDs;

\- presence of every required relevant document;

\- nested subset relationships;

\- consistency with the benchmark manifest.



\---



\## 7. Embedding Model



All V1 experiments use:



```text

sentence-transformers/all-MiniLM-L6-v2

```



Embedding dimension:



```text

384

```



Batch size:



```text

64

```



All document and query embeddings are L2 normalized.



\---



\## 8. Exact Vector Index



V1 uses:



```text

FAISS IndexFlatIP

```



`IndexFlatIP` performs exhaustive exact nearest-neighbor search.



Because embeddings are L2 normalized, inner-product ranking is equivalent to cosine-similarity ranking.



Unlike ANN indexes, `IndexFlatIP` does not introduce approximation error.



Therefore:



> Any change in relevance-based Recall@K observed in V1 is caused by the changing retrieval search space and embedding behavior, not ANN approximation.



\---



\## 9. Exact Ground Truth



For each corpus size, V1 saves the exact Top-10 nearest neighbors for every evaluation query.



Examples of generated artifacts include:



```text

results/v1\_exact\_neighbors\_1k.jsonl

results/v1\_exact\_neighbors\_10k.jsonl

results/v1\_exact\_neighbors\_25k.jsonl

results/v1\_exact\_neighbors\_50k.jsonl

results/v1\_exact\_neighbors\_100k.jsonl

```



Each record contains:



```json

{

&#x20; "query\_id": "...",

&#x20; "neighbors": \[

&#x20;   {

&#x20;     "rank": 1,

&#x20;     "doc\_id": "...",

&#x20;     "score": 0.0

&#x20;   }

&#x20; ],

&#x20; "relevant\_document\_ids": \[

&#x20;   "..."

&#x20; ]

}

```



These exact neighbors will become the algorithmic ground truth for V2 ANN evaluation.



\---



\## 10. Exact-Neighbor Validation



Exact-neighbor artifacts are independently validated by:



```text

src/validate\_exact\_neighbors.py

```



The validator checks:



\- complete benchmark query set;

\- unique query records;

\- exactly 10 neighbors per query;

\- rank sequence 1 through 10;

\- unique neighbors within each query;

\- every returned neighbor belongs to the corresponding corpus;

\- all similarity scores are finite;

\- similarity scores are monotonically non-increasing;

\- relevant-document metadata agrees with qrels.



All final V1 exact-neighbor artifacts pass these checks.



\---



\## 11. SHA-256 Provenance



V1 records cryptographic fingerprints for:



```text

corpus.jsonl

queries.jsonl

qrels.json

benchmark\_manifest.json

```



using:



```text

SHA-256

```



Each exact-search experiment records these fingerprints.



The exact-neighbor validator independently recomputes the hashes and compares them with the experiment metadata.



Conceptually:



```text

Current Benchmark

&#x20;     â†“

SHA-256

&#x20;     â†“

Compare

&#x20;     â†“

Experiment Provenance

&#x20;     â”‚

&#x20;     â”œâ”€â”€ MATCH â†’ continue validation

&#x20;     â”‚

&#x20;     â””â”€â”€ MISMATCH â†’ reject artifact

```



This safeguard prevents exact-neighbor results generated from an earlier benchmark version from being silently reused as ANN ground truth.



The safeguard was explicitly tested using a deliberately corrupted fingerprint.



The corrupted experiment record was correctly rejected.



\---



\## 12. Measurement Methodology



The experiment separates the following stages:



```text

benchmark loading

model loading

document embedding

index construction

query embedding

vector search

```



This separation is important because query-model inference should not be confused with vector-index search latency.



The primary measurements are:



\- document embedding time;

\- embedding throughput;

\- index construction time;

\- serialized index size;

\- query embedding time;

\- mean exact-search latency;

\- median exact-search latency;

\- P95 search latency;

\- P99 search latency;

\- Recall@1;

\- Recall@5;

\- Recall@10;

\- Hit Rate@1;

\- Hit Rate@5;

\- Hit Rate@10;

\- maximum observed Python process RSS.



\---



\## 13. V1 Experimental Results



The final current-provenance results are:



| Documents | Embedding Time (s) | Docs/s | Index Size (MB) | Mean Search (ms) | P95 (ms) | Recall@1 | Recall@5 | Recall@10 | Max RSS (MB) |

|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

| 1,000 | 30.56 | 32.72 | 1.46 | 0.160 | 0.226 | 0.9600 | 0.9967 | 1.0000 | 639.7 |

| 10,000 | 327.05 | 30.58 | 14.65 | 2.625 | 3.603 | 0.9017 | 0.9867 | 0.9933 | 683.4 |

| 25,000 | 856.79 | 29.18 | 36.62 | 4.227 | 5.587 | 0.8683 | 0.9733 | 0.9833 | 766.6 |

| 50,000 | 1,744.18 | 28.67 | 73.24 | 7.874 | 10.114 | 0.8050 | 0.9567 | 0.9800 | 842.7 |

| 100,000 | 3,314.86 | 30.17 | 146.48 | 17.649 | 23.755 | 0.7333 | 0.9300 | 0.9567 | 1,168.8 |



\---



\## 14. Corpus Growth



The corpus grows from:



```text

1,000

```



to:



```text

100,000

```



documents.



This is a:



\\\[

100\\times

\\]



increase in indexed vectors.



\---



\## 15. Embedding-Time Scaling



Document embedding time increases from:



```text

30.56 seconds

```



at 1K documents to:



```text

3,314.86 seconds

```



at 100K documents.



This represents approximately:



\\\[

108.47\\times

\\]



growth in embedding time for a 100Ã— increase in corpus size.



Embedding throughput remains relatively stable:



```text

1K       32.72 docs/s

10K      30.58 docs/s

25K      29.18 docs/s

50K      28.67 docs/s

100K     30.17 docs/s

```



This indicates that embedding cost is approximately linear with the number of documents under the tested single-machine environment.



\---



\## 16. Index-Size Scaling



Serialized `IndexFlatIP` size grows from:



```text

1.46 MB

```



at 1K documents to:



```text

146.48 MB

```



at 100K documents.



This is effectively:



\\\[

100\\times

\\]



growth for a 100Ã— increase in vectors.



The result is expected because `IndexFlatIP` retains the complete float32 representation of every vector.



For 384-dimensional float32 embeddings, the theoretical raw storage per vector is:



\\\[

384 \\times 4 = 1536 \\text{ bytes}

\\]



Therefore, exact flat-vector storage scales linearly with collection size.



\---



\## 17. Exact Search Latency



Mean exact-search latency increases from:



```text

0.160 ms/query

```



at 1K documents to:



```text

17.649 ms/query

```



at 100K documents.



This corresponds to approximately:



\\\[

110.40\\times

\\]



latency growth.



P95 latency increases from:



```text

0.226 ms

```



to:



```text

23.755 ms

```



or approximately:



\\\[

105.34\\times

\\]



growth.



The measured progression is:



```text

Corpus       Mean Search       P95

\----------------------------------------

1K             0.160 ms       0.226 ms

10K            2.625 ms       3.603 ms

25K            4.227 ms       5.587 ms

50K            7.874 ms      10.114 ms

100K          17.649 ms      23.755 ms

```



This demonstrates the central scalability limitation of exhaustive exact vector search.



As the vector collection grows, every query must be compared against an increasingly large number of vectors.



\---



\## 18. Memory Scaling



Maximum observed Python RSS increases from:



```text

639.7 MB

```



at 1K documents to:



```text

1,168.8 MB

```



at 100K documents.



This corresponds to approximately:



\\\[

1.83\\times

\\]



growth.



Process RSS includes more than the FAISS index itself.



It includes components such as:



\- Python runtime;

\- Sentence Transformer model;

\- PyTorch;

\- document text;

\- embedding arrays;

\- query embeddings;

\- FAISS index;

\- experiment data structures.



Therefore, process RSS should not be interpreted as index size.



The serialized FAISS index size is reported separately.



\---



\## 19. Relevance Recall Scaling



Recall decreases as the retrieval search space grows.



\### Recall@1



```text

1K       0.9600

10K      0.9017

25K      0.8683

50K      0.8050

100K     0.7333

```



Overall change:



\\\[

\-0.2267

\\]



\### Recall@5



```text

1K       0.9967

10K      0.9867

25K      0.9733

50K      0.9567

100K     0.9300

```



Overall change:



\\\[

\-0.0667

\\]



\### Recall@10



```text

1K       1.0000

10K      0.9933

25K      0.9833

50K      0.9800

100K     0.9567

```



Overall change:



\\\[

\-0.0433

\\]



\---



\## 20. Why Recall Declines Under Exact Search



This result is important.



`IndexFlatIP` performs exact nearest-neighbor search.



Therefore, the decline in relevance Recall@K is \*\*not caused by approximate indexing\*\*.



Instead, the benchmark introduces more competing documents as corpus size increases.



Conceptually:



```text

Small corpus

&#x20;    â†“

few competing vectors

&#x20;    â†“

relevant document often ranks near top



Larger corpus

&#x20;    â†“

many more competing vectors

&#x20;    â†“

additional semantically similar distractors

&#x20;    â†“

relevant document may move below Top-K

```



This establishes a crucial distinction for future experiments.



\---



\## 21. Relevance Recall vs. ANN Recall



The project distinguishes two different retrieval metrics.



\### Relevance Recall



Relevance Recall asks:



> Does the vector-search system retrieve documents identified as relevant by MS MARCO?



For query \\(q\\):



\\\[

Recall@K\_{relevance}(q)

=

\\frac{

|\\text{Relevant}(q)\\cap TopK(q)|

}{

|\\text{Relevant}(q)|

}

\\]



V1 measures this metric.



\### ANN Recall



Future ANN experiments will additionally ask:



> How closely does an approximate index reproduce the exact nearest neighbors?



For query \\(q\\):



\\\[

Recall@K\_{ANN}(q)

=

\\frac{

|TopK\_{ANN}(q)\\cap TopK\_{Exact}(q)|

}{

K

}

\\]



These metrics answer different questions.



A system can have:



```text

high ANN Recall

```



while still having:



```text

lower relevance Recall

```



if the embedding model itself ranks non-relevant vectors above relevant documents.



\---



\## 22. Why This Distinction Matters for V2



Suppose an ANN index at 100K produces:



```text

ANN Recall@10 = 0.98

```



This would mean that the approximate index reproduced 98% of the exact Top-10 neighbors.



It would \*\*not\*\* mean:



```text

MS MARCO relevance Recall@10 = 0.98

```



The V1 exact-search baseline at 100K already has:



```text

Relevance Recall@10 = 0.9567

```



before any ANN approximation is introduced.



V2 must therefore separate:



```text

embedding/retrieval quality

```



from:



```text

ANN approximation quality

```



\---



\## 23. V1 Scaling Summary



From 1K to 100K documents:



| Measurement | Scaling |

|---|---:|

| Corpus size | 100.00Ã— |

| Embedding time | 108.47Ã— |

| Index size | 100.00Ã— |

| Mean search latency | 110.40Ã— |

| P95 search latency | 105.34Ã— |

| Python RSS | 1.83Ã— |



Retrieval-quality changes:



| Metric | 1K | 100K | Change |

|---|---:|---:|---:|

| Recall@1 | 0.9600 | 0.7333 | -0.2267 |

| Recall@5 | 0.9967 | 0.9300 | -0.0667 |

| Recall@10 | 1.0000 | 0.9567 | -0.0433 |



\---



\## 24. Primary V1 Findings



\### Finding 1 â€” Embedding is the dominant offline cost



Embedding time dominates preprocessing.



At 100K documents:



```text

Embedding time â‰ˆ 3,315 seconds

Index construction â‰ˆ 0.063 seconds

```



The cost of creating document embeddings is orders of magnitude larger than constructing an exact flat index from already-generated vectors.



\---



\### Finding 2 â€” Embedding throughput remains stable



Embedding throughput remains approximately:



```text

29â€“33 documents/second

```



across the tested corpus sizes.



This suggests approximately linear embedding-time scaling under the tested environment.



\---



\### Finding 3 â€” Exact index storage is linear



`IndexFlatIP` storage scales almost perfectly linearly with vector count.



A 100Ã— increase in indexed vectors produces approximately a 100Ã— increase in serialized index size.



\---



\### Finding 4 â€” Exact search latency grows substantially



Mean query latency increases from:



```text

0.160 ms

```



to:



```text

17.649 ms

```



as the corpus grows from 1K to 100K.



This is approximately:



```text

110.4Ã—

```



latency growth.



This result motivates the transition to Approximate Nearest Neighbor indexing.



\---



\### Finding 5 â€” Larger search spaces reduce relevance Recall@K



Recall@5 decreases from:



```text

0.9967

```



at 1K to:



```text

0.9300

```



at 100K.



Because search is exact, this decline reflects the increasing difficulty of semantic retrieval over a larger vector collection rather than ANN approximation error.



\---



\### Finding 6 â€” Exact ground truth is now available



Every V1 scale point has a validated exact Top-10 neighbor artifact.



These artifacts provide the algorithmic reference needed to measure ANN approximation quality in V2.



\---



\### Finding 7 â€” Provenance validation prevents stale ground truth



SHA-256 provenance checks successfully detect when an exact-search artifact was generated against a different benchmark version.



This prevents stale or mismatched exact neighbors from being used as ANN ground truth.



\---



\## 25. Generated Analysis Artifacts



The V1 analysis program is:



```text

src/analyze\_v1\_scaling.py

```



It automatically filters experiment history using the current benchmark-manifest SHA-256.



The cumulative experiment log contained six successful rows:



```text

6 successful experiment records

```



but one historical 1K run belonged to an earlier benchmark manifest.



The analyzer correctly selected:



```text

5 current-provenance runs

```



corresponding to:



```text

1K

10K

25K

50K

100K

```



The generated summary is:



```text

results/v1\_scaling\_summary.csv

```



Generated figures are:



```text

results/v1\_embedding\_time.png

results/v1\_embedding\_throughput.png

results/v1\_index\_size.png

results/v1\_search\_latency.png
