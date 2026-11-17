from ailearn.bm25 import BM25
from ailearn.hybrid import reciprocal_rank_fusion, weighted_score_fusion
from ailearn.llm import MockLLM, ScriptedLLM
from ailearn.rag import RAGPipeline, build_prompt
from ailearn.vectorstore import SearchResult


def test_bm25_prefers_exact_terms(docs):
    bm = BM25(list(docs.values()))
    top_i, _ = bm.search("error 2200 copy activity", k=1)[0]
    assert list(docs)[top_i] == "errors"
    assert bm.search("zzzz-unknown", k=3) == []


def test_rrf_rewards_items_ranked_high_in_both_lists():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    assert [d for d, _ in fused][:2] in (["a", "b"], ["b", "a"])
    assert dict(fused)["d"] < dict(fused)["a"]


def test_weighted_fusion_alpha_extremes():
    dense, sparse = {"a": 0.9, "b": 0.1}, {"a": 1.0, "b": 9.0}
    assert weighted_score_fusion(dense, sparse, alpha=1.0)[0][0] == "a"
    assert weighted_score_fusion(dense, sparse, alpha=0.0)[0][0] == "b"


def test_prompt_contains_numbered_context_and_question():
    p = build_prompt("Q?", [SearchResult("x", 1.0, "ctx one"), SearchResult("y", 0.5, "ctx two")])
    assert "[1] ctx one" in p and "[2] ctx two" in p and "<question>Q?</question>" in p


def test_rag_end_to_end_answers_from_context(docs):
    rag = RAGPipeline(top_k=2)
    assert rag.add_documents(docs) >= len(docs)
    ans = rag.ask("Which layer holds business-level aggregates?")
    assert ans.sources[0].metadata["doc_id"] == "medallion"
    assert "Gold" in ans.answer and "[" in ans.answer  # answer + citation


def test_rag_says_i_dont_know_when_context_irrelevant(docs):
    rag = RAGPipeline(top_k=2, min_score=0.2)
    rag.add_documents(docs)
    ans = rag.ask("Who won the football world cup?")
    assert "don't know" in ans.answer


def test_hybrid_retrieval_finds_exact_error_codes(docs):
    rag = RAGPipeline(top_k=1, hybrid=True)
    rag.add_documents(docs)
    assert rag.retrieve("what does DF-SYS-01 mean")[0].metadata["doc_id"] == "errors"


def test_metadata_filtering_in_rag(docs):
    rag = RAGPipeline(top_k=3)
    rag.add_documents(docs, metadata={k: {"source": "wiki" if k == "rag" else "runbook"} for k in docs})
    res = rag.retrieve("components of RAG", where=lambda m: m["source"] == "runbook")
    assert all(r.metadata["doc_id"] != "rag" for r in res)


def test_llm_receives_the_retrieved_context(docs):
    llm = ScriptedLLM(["stub answer"])
    rag = RAGPipeline(llm=llm, top_k=1)
    rag.add_documents(docs)
    rag.ask("What does OPTIMIZE do in Delta?")
    assert "OPTIMIZE" in llm.calls[0]


def test_mock_llm_without_context():
    assert "don't know" in MockLLM().complete("hello")
