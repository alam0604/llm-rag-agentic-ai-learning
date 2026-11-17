"""Best-effort link checker for Markdown files.

    python tools/check_links.py                 # all tracked .md files
    python tools/check_links.py --changed       # only .md files changed vs origin/main / last commit
    python tools/check_links.py docs/04-rag.md  # specific files

Checks relative links (file must exist) always; checks http(s) links when the network allows.
Some sites block bots (403/429) -> reported as warnings, not failures.
Exit code 1 only for missing relative files or hard 404s.
"""
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
FENCE = re.compile(r"```.*?```", re.S)
INLINE = re.compile(r"`[^`\n]*`")


def md_files(args):
    if "--changed" in args:
        out = subprocess.run(["git", "diff", "--name-only", "HEAD~1"], cwd=ROOT, capture_output=True, text=True).stdout
        out += subprocess.run(["git", "diff", "--name-only"], cwd=ROOT, capture_output=True, text=True).stdout
        return sorted({ROOT / f for f in out.split() if f.endswith(".md") and (ROOT / f).exists()})
    files = [a for a in args if a.endswith(".md")]
    if files:
        return [Path(f).resolve() for f in files]
    out = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True).stdout
    return [ROOT / f for f in out.split()]


def check_url(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0 link-check"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status
    except urllib.error.HTTPError as e:
        if e.code in (403, 405, 429, 999):
            return e.code
        if e.code == 404:
            return 404
        return e.code
    except Exception:
        return None  # network blocked / timeout -> unknown


def main(args):
    errors, warnings, seen = [], [], {}
    for f in md_files(args):
        text = INLINE.sub("", FENCE.sub("", f.read_text(encoding="utf-8")))   # ignore code
        for target in LINK.findall(text):
            if target.startswith(("mailto:", "#")):
                continue
            if target.startswith(("http://", "https://")):
                if "--no-web" in args:
                    continue
                status = seen.setdefault(target, check_url(target))
                if status == 404:
                    errors.append(f"{f.relative_to(ROOT)}: 404 {target}")
                elif status is None or status >= 400:
                    warnings.append(f"{f.relative_to(ROOT)}: {status or 'unreachable'} {target}")
            else:
                path = (f.parent / target.split("#")[0]).resolve()
                if target.split("#")[0] and not path.exists():
                    errors.append(f"{f.relative_to(ROOT)}: missing file {target}")
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print(f"{len(errors)} errors, {len(warnings)} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
