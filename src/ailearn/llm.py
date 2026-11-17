"""LLM interface.

Every component in this repo talks to an `LLM` with one method:
    complete(prompt: str) -> str

* MockLLM            — deterministic, offline, used by tests. It "answers" by
                       extracting the context sentence that best overlaps the question,
                       which is enough to demonstrate RAG wiring and grounding.
* ScriptedLLM        — returns pre-written responses in order (great for testing agents).
* OpenAICompatibleLLM — works with OpenAI, Azure OpenAI, Ollama, vLLM, LM Studio, Groq...
* AnthropicLLM       — Claude via the Anthropic SDK.
"""
from __future__ import annotations

import os
import re
from typing import Protocol

from .embeddings import tokenize


class LLM(Protocol):
    def complete(self, prompt: str) -> str: ...


class MockLLM:
    """Offline stand-in: picks the context sentence with highest word overlap with the question."""

    NO_ANSWER = "I don't know based on the provided context."

    def __init__(self):
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        ctx = re.search(r"<context>(.*?)</context>", prompt, re.S)
        q = re.search(r"<question>(.*?)</question>", prompt, re.S)
        if not ctx or not q:
            return self.NO_ANSWER
        q_toks = set(tokenize(q.group(1)))
        best, best_score = None, 0
        for block in re.findall(r"\[(\d+)\](.*?)(?=\n\[\d+\]|\Z)", ctx.group(1), re.S):
            cid, text = block
            for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
                score = len(q_toks & set(tokenize(sent)))
                if score > best_score:
                    best, best_score = (sent.strip(), cid), score
        if not best or best_score < 1:
            return self.NO_ANSWER
        return f"{best[0]} [{best[1]}]"


class ScriptedLLM:
    def __init__(self, responses: list[str]):
        self.responses, self.calls = list(responses), []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        if not self.responses:
            raise RuntimeError("ScriptedLLM ran out of responses")
        return self.responses.pop(0)


class OpenAICompatibleLLM:  # pragma: no cover - needs network + key
    """pip install openai. For Ollama: base_url='http://localhost:11434/v1', api_key='ollama'."""

    def __init__(self, model: str = "gpt-4o-mini", base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.0):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model, self.temperature = model, temperature

    def complete(self, prompt: str) -> str:
        r = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content


class AnthropicLLM:  # pragma: no cover - needs network + key
    """pip install anthropic; export ANTHROPIC_API_KEY=..."""

    def __init__(self, model: str = "claude-sonnet-4-5", max_tokens: int = 1024):
        import anthropic
        self.client, self.model, self.max_tokens = anthropic.Anthropic(), model, max_tokens

    def complete(self, prompt: str) -> str:
        r = self.client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                        messages=[{"role": "user", "content": prompt}])
        return r.content[0].text
