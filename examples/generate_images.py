"""Regenerate every picture in assets/images from the code in this repo.

    python examples/generate_images.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ailearn.attention import causal_mask, scaled_dot_product_attention, softmax  # noqa: E402
from ailearn.embeddings import HashingEmbedder  # noqa: E402
from ailearn.vectorstore import kmeans  # noqa: E402

OUT = ROOT / "assets" / "images"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = "#2a6fdb", "#f28e2b", "#3a9d5d", "#d64545", "#8a5cd1", "#888888"


def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name, bbox_inches="tight")
    plt.close()
    print("wrote", name)


# 1. attention heatmap (causal) --------------------------------------------
tokens = ["The", "pipeline", "failed", "because", "it", "ran", "out", "of", "memory"]
rng = np.random.default_rng(3)
x = rng.normal(size=(len(tokens), 16))
x[4] = x[1] * 1.6 + rng.normal(size=16) * 0.2  # make "it" resemble "pipeline"
_, w = scaled_dot_product_attention(x, x, x, causal_mask(len(tokens)))
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(w, cmap="Blues")
ax.set_xticks(range(len(tokens)), tokens, rotation=45, ha="right")
ax.set_yticks(range(len(tokens)), tokens)
ax.set_xlabel("attends to (keys)")
ax.set_ylabel("query token")
ax.set_title("Causal self-attention weights\n(upper triangle = future = masked)")
fig.colorbar(im, ax=ax, shrink=0.8)
save("attention_heatmap.png")

# 2. temperature effect --------------------------------------------------------
words = ["Spark", "Python", "SQL", "Java", "Excel", "COBOL"]
logits = np.array([3.0, 2.5, 2.2, 1.0, 0.5, -1.0])
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
for ax, t, c in zip(axes, [0.3, 1.0, 2.0], [BLUE, GREEN, ORANGE]):
    ax.bar(words, softmax(logits / t), color=c)
    ax.set_title(f"temperature = {t}")
    ax.tick_params(axis="x", rotation=40)
axes[0].set_ylabel("P(next token)")
fig.suptitle('"The best language for big-data ETL is ___"   — low T = focused, high T = creative/random')
save("temperature.png")

# 3. embedding space (2D projection) ------------------------------------------
sentences = {
    "Data engineering": ["build an ETL pipeline in spark", "ingest files with azure data factory",
                         "optimize delta table with zorder", "schedule a databricks job", "partition parquet files by date"],
    "LLM / RAG": ["retrieve chunks from a vector database", "embed documents for semantic search",
                  "prompt the llm with retrieved context", "rerank retrieved passages", "reduce hallucination with rag"],
    "Cooking": ["bake bread with yeast and flour", "fry onions until golden", "boil pasta in salted water",
                "season the soup with pepper", "knead the dough for ten minutes"],
}
emb = HashingEmbedder(dim=256, use_bigrams=False)
labels, texts = zip(*[(k, s) for k, v in sentences.items() for s in v])
# add shared words so clusters are visible with a lexical embedder
texts = [t + " " + {"Data engineering": "data pipeline etl", "LLM / RAG": "llm retrieval embedding",
                    "Cooking": "kitchen recipe food"}[l] for t, l in zip(texts, labels)]
v = emb.embed(list(texts))
v = v - v.mean(0)
_, _, vt = np.linalg.svd(v, full_matrices=False)
p = v @ vt[:2].T
fig, ax = plt.subplots(figsize=(7, 5))
for lab, c in zip(sentences, [BLUE, PURPLE, ORANGE]):
    idx = [i for i, l in enumerate(labels) if l == lab]
    ax.scatter(p[idx, 0], p[idx, 1], s=70, c=c, label=lab, alpha=0.85)
ax.legend()
ax.set_title("Embedding space: similar meaning → nearby points\n(PCA projection of text embeddings)")
ax.set_xticks([]); ax.set_yticks([])
save("embedding_space.png")

# 4. similarity metrics --------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 5))
q, a, b = np.array([2, 1]), np.array([4, 2.2]), np.array([1.2, 1.6])
for vec, name, c in [(q, "query", RED), (a, "doc A (same direction, longer)", BLUE), (b, "doc B (closer, different angle)", GREEN)]:
    ax.annotate("", xy=vec, xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=2.2, color=c))
    ax.text(vec[0] + 0.08, vec[1] + 0.08, name, color=c)
ax.set_xlim(0, 5.5); ax.set_ylim(0, 3)
cos = lambda u, w: u @ w / np.linalg.norm(u) / np.linalg.norm(w)  # noqa: E731
ax.set_title(f"cosine(q,A)={cos(q,a):.2f}  cosine(q,B)={cos(q,b):.2f}\n"
             f"euclid(q,A)={np.linalg.norm(q-a):.2f}  euclid(q,B)={np.linalg.norm(q-b):.2f}")
save("similarity_metrics.png")

# 5. IVF clustering --------------------------------------------------------------
rng = np.random.default_rng(7)
centers = np.array([[2, 2], [-2, 2], [-2, -2], [2, -2], [0, 3.5]])
pts = np.vstack([c + rng.normal(size=(60, 2)) * 0.55 for c in centers])
cent = centers.astype(float) + rng.normal(size=centers.shape) * 0.3  # euclidean k-means for the 2-D picture
for _ in range(15):
    assign = np.argmin(np.linalg.norm(pts[:, None] - cent[None], axis=2), axis=1)
    cent = np.array([pts[assign == i].mean(0) for i in range(5)])
qpt = np.array([1.4, 2.6])
probe = np.argsort(np.linalg.norm(cent - qpt, axis=1))[:2]
fig, ax = plt.subplots(figsize=(6.5, 5.5))
cols = [BLUE, GREEN, ORANGE, PURPLE, GREY]
for i in range(5):
    m = assign == i
    ax.scatter(pts[m, 0], pts[m, 1], s=14, c=cols[i], alpha=0.9 if i in probe else 0.18)
    ax.scatter(*cent[i], marker="X", s=180, c="black")
ax.scatter(*qpt, marker="*", s=380, c=RED, label="query")
ax.legend(loc="lower left")
ax.set_title("IVF index: only scan the n_probe=2 nearest clusters\n(faded clusters are skipped → much faster, slightly less exact)")
save("ivf_clusters.png")

# 6. HNSW layers ------------------------------------------------------------------
rng = np.random.default_rng(11)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
allp = rng.uniform(0, 10, size=(40, 2))
layers = [allp[:4], allp[:12], allp]
for li, (ax, P) in enumerate(zip(axes, layers)):
    for i in range(len(P)):
        d = np.linalg.norm(P - P[i], axis=1)
        for j in np.argsort(d)[1:4]:
            ax.plot(*zip(P[i], P[j]), c=GREY, lw=0.6, alpha=0.6)
    ax.scatter(P[:, 0], P[:, 1], s=30, c=[BLUE, GREEN, ORANGE][li])
    path = [P[0], P[min(2, len(P)-1)], P[min(3, len(P)-1)]]
    ax.plot(*zip(*path), c=RED, lw=2.2)
    ax.set_title(["Layer 2 (sparse: long jumps)", "Layer 1", "Layer 0 (all vectors: fine search)"][li])
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("HNSW: greedy search starts at the top sparse layer and descends to dense layers")
save("hnsw_layers.png")

# 7. chunk size trade-off ----------------------------------------------------------
sizes = np.array([64, 128, 256, 512, 1024, 2048])
precision = np.array([0.55, 0.68, 0.78, 0.80, 0.72, 0.6])
context = np.array([0.35, 0.55, 0.75, 0.88, 0.95, 0.98])
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(sizes, precision, "o-", c=BLUE, label="retrieval precision (focused embedding)")
ax.plot(sizes, context, "s-", c=ORANGE, label="context completeness per chunk")
ax.axvspan(256, 1024, color=GREEN, alpha=0.12, label="common sweet spot")
ax.set_xscale("log", base=2); ax.set_xticks(sizes, sizes)
ax.set_xlabel("chunk size (tokens)"); ax.set_ylabel("illustrative score")
ax.set_title("Chunk size trade-off (illustrative) — always tune on YOUR eval set")
ax.legend(fontsize=8)
save("chunk_size_tradeoff.png")

print("done")
