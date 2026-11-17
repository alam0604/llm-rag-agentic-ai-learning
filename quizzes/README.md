# 📝 Quizzes & Interview Questions

Answer before expanding. Each module doc also ends with its own "Check your understanding" section.

## 🟢 Level 1 — Basics

<details><summary>1. What does an LLM fundamentally do?</summary>Predicts a probability distribution over the next token given previous tokens, repeatedly (autoregressive generation).</details>
<details><summary>2. What is a token? Roughly how many tokens in 1,000 English words?</summary>A sub-word unit produced by a tokenizer (e.g. BPE). ≈ 1,300–1,400 tokens.</details>
<details><summary>3. What is a context window?</summary>The maximum number of tokens (input + output) a model can process in one request.</details>
<details><summary>4. What is an embedding?</summary>A fixed-length vector representing meaning; similar meanings → nearby vectors.</details>
<details><summary>5. Cosine similarity of [1,0] and [0,1]? Of [1,1] and [2,2]?</summary>0 (orthogonal); 1 (same direction).</details>
<details><summary>6. What is temperature?</summary>A divisor on logits before softmax: low → deterministic/focused, high → diverse/random.</details>
<details><summary>7. What is a hallucination and one way to reduce it?</summary>Fluent but false/unsupported output. Reduce with RAG + grounding instructions + citations + "I don't know" option.</details>
<details><summary>8. What does RAG stand for and what are its two phases?</summary>Retrieval-Augmented Generation; indexing (chunk → embed → store) and querying (retrieve → augment prompt → generate).</details>
<details><summary>9. What is a vector database?</summary>A system that stores vectors + metadata and supports fast (approximate) nearest-neighbour search with filtering, persistence, scaling.</details>
<details><summary>10. Define an AI agent.</summary>An LLM that runs in a loop, choosing tools/actions based on observations to achieve a goal.</details>
<details><summary>11. System prompt vs user prompt?</summary>System sets role/rules/format for the whole conversation; user messages carry the requests.</details>
<details><summary>12. Few-shot prompting?</summary>Including a few input→output examples in the prompt to demonstrate the task/format.</details>

## 🟡 Level 2 — Intermediate

<details><summary>13. Explain Q, K, V in self-attention.</summary>Query = what a token looks for; Key = what a token offers to be matched; Value = the information passed along. Output = softmax(QKᵀ/√d)·V.</details>
<details><summary>14. Why is chunk overlap used?</summary>So information spanning a boundary appears intact in at least one chunk.</details>
<details><summary>15. Bi-encoder vs cross-encoder?</summary>Bi-encoder embeds query and doc separately (fast, ANN-able, for retrieval). Cross-encoder scores the pair jointly (accurate, slow, for reranking top-N).</details>
<details><summary>16. What is HNSW and its main tuning knobs?</summary>A layered proximity graph for ANN search; M (edges/node), ef_construction (build quality), ef_search (query recall vs latency).</details>
<details><summary>17. What is hybrid search and how is RRF computed?</summary>Combining dense and keyword (BM25) retrieval; RRF score = Σ 1/(k + rank) across result lists (k≈60).</details>
<details><summary>18. Recall@k vs precision@k vs MRR?</summary>Recall@k: fraction of relevant docs found in top k. Precision@k: fraction of top k that is relevant. MRR: mean of 1/rank of the first relevant result.</details>
<details><summary>19. What is "lost in the middle"?</summary>LLMs use information at the start and end of long contexts better than in the middle → put the best chunks first/last, use fewer chunks.</details>
<details><summary>20. Explain the ReAct pattern.</summary>Interleave Thought → Action (tool call) → Observation until a Final Answer.</details>
<details><summary>21. Why should a SQL tool for an agent be read-only?</summary>Model outputs are untrusted (errors, prompt injection); least privilege prevents destructive changes.</details>
<details><summary>22. Pre-filtering vs post-filtering in vector search?</summary>Pre: restrict candidates by metadata before ANN (complete results). Post: filter the ANN top-k afterwards (may return fewer/no results).</details>
<details><summary>23. How do you test LLM code deterministically?</summary>Inject mock/scripted LLMs; test tools/parsers as normal functions; temperature 0 and recorded fixtures for integration tests.</details>
<details><summary>24. When choose text-to-SQL over RAG?</summary>For numeric/aggregate questions over structured tables.</details>

## 🔴 Level 3 — Advanced

<details><summary>25. Explain LoRA and why it's efficient.</summary>Freeze W, learn a low-rank update ΔW = BA (rank r ≪ d). Trains ~0.1–1% of params, small adapters, swappable per task.</details>
<details><summary>26. What is the KV cache and why does it limit batch size?</summary>Cached keys/values of past tokens to avoid recomputation; memory grows with layers × seq_len × batch, consuming GPU memory.</details>
<details><summary>27. Describe Corrective RAG.</summary>Grade retrieved docs; if irrelevant/ambiguous, rewrite query or use another source (e.g. web) before generating.</details>
<details><summary>28. GraphRAG vs vector RAG?</summary>GraphRAG extracts entities/relations into a graph + community summaries; better for multi-hop and global questions; costlier to build.</details>
<details><summary>29. Indirect prompt injection — example and defences.</summary>Malicious instructions inside retrieved docs/web pages/tool output. Defences: treat as data, privilege separation, approvals for sensitive tools, filtering, monitoring.</details>
<details><summary>30. How to evaluate an agent?</summary>Task success rate on a benchmark, trajectory quality (right tools/order), cost & steps, safety violations; unit tests with scripted LLMs; LLM-as-judge with rubrics; human review.</details>
<details><summary>31. How would you keep a RAG index fresh over a Delta table?</summary>Enable Change Data Feed; incrementally upsert/delete changed chunks (e.g. Databricks Vector Search Delta Sync); store version/modified dates; re-embed all on model change.</details>
<details><summary>32. Product quantization in one paragraph.</summary>Split vectors into m sub-vectors, cluster each subspace into 256 centroids, store 1-byte codes → huge compression; distances approximated via lookup tables; often IVF-PQ with re-ranking.</details>
<details><summary>33. When fine-tune instead of RAG?</summary>For behaviour/format/style/narrow skills or distilling to a cheaper model — not for frequently changing knowledge.</details>
<details><summary>34. How does speculative decoding speed up inference?</summary>A small draft model proposes several tokens; the large model verifies them in one forward pass and accepts the matching prefix — same output distribution, fewer large-model steps.</details>

## 🎤 System-design interview prompts (practice out loud)
1. Design an internal "ask the data platform" assistant for 2,000 employees with document-level permissions.
2. Design a text-to-SQL system over 800 tables with consistent business metrics.
3. Your RAG bot's answer quality dropped after a re-index. How do you investigate?
4. Design an agent that can remediate failed pipelines automatically — what must require human approval?
5. Reduce the cost of an LLM feature by 80% without losing quality — options?
