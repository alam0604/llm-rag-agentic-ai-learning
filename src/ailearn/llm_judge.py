"""LLM-as-a-judge + an evaluation runner with a CI regression gate.

The pattern behind RAGAS, DeepEval, MLflow judges, promptfoo…:
  1. a golden dataset (inputs + expectations)
  2. run your app on every row
  3. score each output with code metrics and/or an LLM judge using a strict rubric
  4. aggregate, compare with thresholds / the previous run, and fail CI on regression
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from pydantic import BaseModel, Field

from .evaluation import groundedness, token_f1
from .llm import LLM
from .structured import generate_structured


class JudgeVerdict(BaseModel):
    score: int = Field(ge=1, le=5, description="1 = fails the criterion, 5 = fully meets it")
    reasoning: str = Field(description="one or two sentences; quote the problematic claim if any")


RUBRICS = {
    "faithfulness": (
        "Is EVERY claim in the ANSWER supported by the CONTEXT? 5 = all claims supported; "
        "3 = minor unsupported detail; 1 = main claim unsupported or contradicts the context."),
    "answer_relevance": (
        "Does the ANSWER directly address the QUESTION? 5 = fully and concisely; 3 = partially; 1 = off-topic."),
    "correctness": (
        "Compared with the REFERENCE answer, is the ANSWER factually correct? 5 = equivalent; "
        "3 = partially correct; 1 = wrong."),
}


def llm_judge(judge: LLM, criterion: str, question: str, answer: str, context: str = "",
              reference: str = "") -> JudgeVerdict:
    task = (f"You are a strict evaluator. Criterion: {criterion}\nRubric: {RUBRICS[criterion]}\n\n"
            f"QUESTION: {question}\nCONTEXT: {context or '(none)'}\nREFERENCE: {reference or '(none)'}\n"
            f"ANSWER: {answer}\n\nThink about the rubric, then give the score.")
    return generate_structured(judge, task, JudgeVerdict, max_retries=1)


@dataclass
class EvalRow:
    question: str
    answer: str
    context: str
    reference: str
    latency_s: float
    scores: dict[str, float] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)


@dataclass
class EvalReport:
    rows: list[EvalRow]
    metrics: dict[str, float]
    failures: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_json(self) -> str:
        return json.dumps({"metrics": self.metrics, "failures": self.failures,
                           "rows": [asdict(r) for r in self.rows]}, indent=2)

    def to_markdown(self) -> str:
        lines = ["| metric | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in self.metrics.items()]
        if self.failures:
            lines += ["", "**Regressions:** " + "; ".join(self.failures)]
        return "\n".join(lines)


def run_eval(app: Callable[[str], tuple[str, str]], dataset: list[dict], judge: LLM | None = None,
             criteria: tuple[str, ...] = ("faithfulness", "answer_relevance"),
             thresholds: dict[str, float] | None = None, baseline: dict[str, float] | None = None,
             max_drop: float = 0.05, clock: Callable[[], float] = time.perf_counter) -> EvalReport:
    """app(question) -> (answer, context). dataset rows: {"question", "reference"?}."""
    rows = []
    for item in dataset:
        t0 = clock()
        answer, context = app(item["question"])
        row = EvalRow(item["question"], answer, context, item.get("reference", ""), clock() - t0)
        # cheap deterministic metrics
        row.scores["groundedness_lexical"] = groundedness(answer, [context]) if context else 0.0
        if row.reference:
            row.scores["token_f1"] = token_f1(answer, row.reference)
        # LLM judges (normalised to 0..1)
        if judge:
            for c in criteria:
                if c == "correctness" and not row.reference:
                    continue
                v = llm_judge(judge, c, row.question, answer, context, row.reference)
                row.scores[c] = (v.score - 1) / 4
                row.notes[c] = v.reasoning
        rows.append(row)

    keys = sorted({k for r in rows for k in r.scores})
    metrics = {k: round(statistics.mean(r.scores[k] for r in rows if k in r.scores), 4) for k in keys}
    lat = sorted(r.latency_s for r in rows)
    metrics["latency_p50_s"] = round(lat[len(lat) // 2], 4) if lat else 0.0

    failures = []
    for k, t in (thresholds or {}).items():
        if metrics.get(k, 0.0) < t:
            failures.append(f"{k}={metrics.get(k, 0.0)} < threshold {t}")
    for k, prev in (baseline or {}).items():
        if k in metrics and not k.startswith("latency") and metrics[k] < prev - max_drop:
            failures.append(f"{k} dropped {prev} -> {metrics[k]}")
    return EvalReport(rows, metrics, failures)
