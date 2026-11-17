# Instructions for Claude sessions working on this repo

This is Alam's private LLM / RAG / Agentic AI learning repo. It evolves weekly.

- Before adding or changing learning content, read **EVOLUTION.md** (the quality contract and weekly procedure) and **BACKLOG.md** (what to build next).
- Topics under "🔥 Requested" in BACKLOG.md come from Alam. Build them first and honour any notes on the line.
- Keep every test offline: inject `ScriptedLLM` / `MockLLM`, no network or API keys in `tests/`.
- Verify library APIs against current official docs before writing code; never invent URLs.
- Run `pytest -q`, `node tools/check_mermaid.mjs $(git ls-files '*.md')` (after `cd tools && npm install`) and `python tools/check_links.py --changed` before every push. Never push a failing build.
- Log every change in CHANGELOG.md.
- Commit as `alam0604 <alam0604@users.noreply.github.com>` and include the Co-Authored-By trailer for Claude.
