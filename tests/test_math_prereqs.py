"""LEARNING TESTS for docs/deep-dives/07-prerequisites-math-and-python.md.
Every assertion is a fact you should be able to explain."""
import numpy as np
import pytest
from pydantic import BaseModel, Field, ValidationError, field_validator


def test_dot_product_by_hand():
    q, d = np.array([1, 2, 0]), np.array([2, 3, 1])
    assert q @ d == 1 * 2 + 2 * 3 + 0 * 1 == 8


def test_norm_and_cosine_by_hand():
    q, d = np.array([1, 2, 0]), np.array([2, 3, 1])
    assert np.linalg.norm(q) == pytest.approx(5 ** 0.5)
    cos = q @ d / (np.linalg.norm(q) * np.linalg.norm(d))
    assert cos == pytest.approx(0.956, abs=1e-3)


def test_normalized_vectors_make_dot_equal_cosine():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=8), rng.normal(size=8)
    an, bn = a / np.linalg.norm(a), b / np.linalg.norm(b)
    assert an @ bn == pytest.approx(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_matrix_multiply_shapes_are_how_search_works():
    docs = np.random.default_rng(1).normal(size=(1000, 384))   # 1000 chunks, 384-dim embeddings
    query = np.random.default_rng(2).normal(size=(384,))
    scores = docs @ query                                       # (1000,384) @ (384,) -> (1000,)
    assert scores.shape == (1000,)
    top3 = np.argsort(-scores)[:3]
    assert scores[top3[0]] == scores.max()


def test_softmax_turns_scores_into_probabilities():
    z = np.array([2.0, 1.0, 0.1])
    p = np.exp(z) / np.exp(z).sum()
    assert p.sum() == pytest.approx(1.0) and p.argmax() == 0
    assert p.round(2).tolist() == [0.66, 0.24, 0.1]


def test_memory_math_for_vector_index():
    n, dim, bytes_per_float = 10_000_000, 1536, 4
    assert n * dim * bytes_per_float / 1e9 == pytest.approx(61.44)   # GB of raw float32 vectors


class TableDoc(BaseModel):
    table: str = Field(pattern=r"^[a-z_]+\.[a-z_]+$")
    owner: str
    sla_hours: int = Field(gt=0, le=168)
    pii_columns: list[str] = []

    @field_validator("pii_columns")
    @classmethod
    def lowercase(cls, v):
        return [c.lower() for c in v]


def test_pydantic_validates_coerces_and_rejects():
    doc = TableDoc.model_validate({"table": "gold.sales", "owner": "alam", "sla_hours": "24", "pii_columns": ["EMAIL"]})
    assert doc.sla_hours == 24 and doc.pii_columns == ["email"]       # "24" coerced to int; validator ran
    with pytest.raises(ValidationError) as e:
        TableDoc(table="Gold Sales", owner="x", sla_hours=500)
    assert {err["loc"][0] for err in e.value.errors()} == {"table", "sla_hours"}
    assert TableDoc.model_json_schema()["properties"]["sla_hours"]["maximum"] == 168  # schema for LLMs
