# 📋 Topic Backlog — what the repo learns next

This file drives the repo's **weekly automatic update**. Each run takes the **top unchecked item**, builds it to the standard in [EVOLUTION.md](EVOLUTION.md), ticks it off here, and logs it in [CHANGELOG.md](CHANGELOG.md).

**✍️ You steer it:** edit this file on GitHub (pencil icon) at any time.
- Add a line under **🔥 Requested** to get that topic next.
- Reorder lines to change priority, or delete topics you don't care about.
- Write a note after a topic, e.g. `— focus on Fabric, include a PySpark example`.

## 🔥 Requested (always built first, top to bottom)
<!-- Add your own topics here, one per line, e.g.:
- [ ] Text-to-SQL on Microsoft Fabric with data agents — compare with Databricks Genie
-->

## 🧭 Planned (built in this order)
- [ ] **Multi-agent supervisor in LangGraph** — supervisor + SQL/docs/ops sub-agents, handoffs, scripted-LLM tests (`src/ailearn/supervisor.py`)
- [ ] **Text-to-SQL deep dive** — semantic layer / metric definitions, schema retrieval for 100s of tables, execution-accuracy eval harness with a public dataset
- [ ] **Guardrails & prompt-injection defense** — input/output filters, injection test suite of malicious docs, Llama Guard / content-safety APIs, code + tests
- [ ] **Semantic caching** — embedding-similarity cache with TTL and ACL scoping, `LLM`-interface class + tests
- [ ] **GraphRAG hands-on** — build a lineage/knowledge graph with NetworkX from table metadata, multi-hop questions vs vector RAG, code + tests
- [ ] **Microsoft Fabric + AI** — AI functions, Copilot, data agents, RAG over OneLake; compare with Databricks
- [ ] **Context engineering & prompt caching** — what goes in the context window, ordering, compression, caching economics, worked cost examples
- [ ] **Reasoning models & test-time compute** — when to use them, prompting differences, cost/latency trade-offs, evaluation
- [ ] **Agent memory systems** — short/long-term, LangGraph Store with semantic search, summarisation, forgetting, privacy
- [ ] **Fine-tuning hands-on (LoRA/QLoRA)** — tiny model on a classification task, data prep, eval vs prompting, Colab-friendly notebook
- [ ] **Local & small models** — Ollama/llama.cpp/vLLM, quantization formats (GGUF/AWQ), benchmarking quality vs speed on a laptop
- [ ] **Multimodal RAG** — PDFs with tables/images, page-image retrieval (ColPali-style), captioning, Document Intelligence
- [ ] **Streaming RAG ingestion** — Auto Loader → Delta (CDF) → Vector Search sync; deletes & updates; data-quality checks on chunks
- [ ] **Vector DB benchmark project** — pgvector vs Qdrant vs Chroma: recall@10, latency, memory on the same dataset (docker-compose)
- [ ] **Agent evaluation benchmarks** — task success, trajectory evals, popular public benchmarks and what they measure
- [ ] **Cost engineering playbook** — routing/cascades, batch APIs, caching, token budgets, a cost dashboard example
- [ ] **AI system-design interview walkthroughs** — 5 full worked designs (enterprise RAG, text-to-SQL, triage agent, doc extraction, support bot)

## ✅ Done
<!-- The weekly run moves finished items here with the date and a link. -->
