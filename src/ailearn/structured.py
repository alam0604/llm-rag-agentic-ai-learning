"""Reliable structured output with Pydantic: parse -> validate -> retry with the error.

Use native structured outputs (OpenAI `response_format`, Anthropic tool schemas, Ollama
`format=`) when your provider supports them. This helper is the portable fallback, and it
shows exactly what those features protect you from.
"""
from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .llm import LLM

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    def __init__(self, msg: str, attempts: list[str]):
        super().__init__(msg)
        self.attempts = attempts


def extract_json(text: str) -> str:
    """Pull a JSON object/array out of chatty model output (```json fences, leading prose…)."""
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    if fence:
        return fence.group(1).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return text.strip()
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    return text[start:end + 1] if end > start else text[start:]


def schema_prompt(model: type[BaseModel]) -> str:
    return ("Respond with ONLY a JSON object that matches this JSON Schema (no prose, no markdown):\n"
            + json.dumps(model.model_json_schema(), indent=2))


def generate_structured(llm: LLM, task: str, model: type[T], max_retries: int = 2) -> T:
    """Ask for JSON matching `model`; on parse/validation failure, show the model its error and retry."""
    prompt = f"{task}\n\n{schema_prompt(model)}"
    raw_outputs: list[str] = []
    for _ in range(max_retries + 1):
        raw = llm.complete(prompt)
        raw_outputs.append(raw)
        try:
            return model.model_validate_json(extract_json(raw))
        except (ValidationError, ValueError) as e:
            err = str(e).splitlines()
            short = "\n".join(err[:8])
            prompt = (f"{task}\n\n{schema_prompt(model)}\n\nYour previous answer was invalid:\n{raw}\n\n"
                      f"Validation errors:\n{short}\nReturn corrected JSON only.")
    raise StructuredOutputError(f"no valid {model.__name__} after {max_retries + 1} attempts", raw_outputs)
