"""Select bounded, verbatim excerpts with useful local context."""

import ast
import re


MAX_QUOTE = 240


def source_scope(path):
    """File role hints help distinguish a sample app from library internals."""
    parts = path.lower().replace("\\", "/").split("/")
    if parts[-1].endswith(".d.ts"):
        return "Type declarations, possibly test-only; not runtime implementation. Read comments and imports for scope."
    if any(p in {"test", "tests", "examples", "example", "demo", "demos"}
           or p.startswith("e2e") for p in parts[:-1]) or parts[-1].startswith("test_"):
        return "Test/example code; not proof of library implementation or overall coverage."
    if parts[:2] == [".github", "workflows"]:
        return "Workflow configuration; only visible triggers and steps are established."
    if parts[-1].startswith("readme") or "docs" in parts:
        return "Documentation; distinguish stated features from verified runtime behavior."
    return "Sampled file; do not generalize beyond the visible content."


def _prefix(text):
    """Keep complete lines when a block is larger than the quote budget."""
    text = text.strip()
    if len(text) <= MAX_QUOTE:
        return text
    end = text.rfind("\n", 0, MAX_QUOTE + 1)
    if end >= 12:
        return text[:end].rstrip()
    # Prose may be one long line. Do not cut through a word.
    end = text.rfind(" ", 0, MAX_QUOTE + 1)
    return text[:end if end >= 12 else MAX_QUOTE].rstrip()


def _decorative(text):
    return bool(re.search(r"<[/]?(?:p|a|img|picture|source|div)\b|!\[|shields\.io|badge", text, re.I))


def _candidates(text, path):
    """Return (priority, quote, partial) entries; never synthesize source text."""
    lines = text.splitlines(keepends=True)
    candidates = []
    if path.endswith(".py"):
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            tree = None  # The inspected file may have been truncated.
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    block = "".join(lines[node.lineno - 1:node.end_lineno]).strip()
                    quote = _prefix(block)
                    complete = len(block) <= MAX_QUOTE
                    # Small complete functions establish behavior better than imports.
                    priority = 10 if complete else 6
                    if isinstance(node, ast.ClassDef):
                        priority -= 2
                    candidates.append((priority, quote, not complete))
        if candidates:
            return candidates
        # Keep the function name attached to the visible body even in partial files.
        for i, line in enumerate(lines):
            if re.match(r"\s*(?:async\s+)?(?:def|class)\s+", line):
                candidates.append((5, _prefix("".join(lines[i:])), True))
        if candidates:
            return candidates

    config = path.endswith((".toml", ".yml", ".yaml", ".ini", ".cfg"))
    if config:
        for i, line in enumerate(lines):
            if re.match(r'\s*["\'][A-Za-z][A-Za-z0-9_.-]*(?:\[.*?\])?\s*[<>=!~]', line):
                candidates.append((9, _prefix(line), False))
        for i, line in enumerate(lines):
            if re.match(r"\s*(?:\[|on\s*:|(?:-\s*)?run\s*:|requires-python\s*=|dependencies\s*=|build-backend\s*=)", line):
                end = i + 1
                # Include neighboring settings/steps but stay within a bounded window.
                while end < len(lines) and lines[end].strip() and len("".join(lines[i:end + 1]).strip()) <= MAX_QUOTE:
                    end += 1
                block = "".join(lines[i:end]).strip()
                candidates.append((8, _prefix(block), True))
        if candidates:
            return candidates

    # Paragraph boundaries avoid mixing badge markup with explanatory prose.
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block or _decorative(block):
            continue
        if (len(block) < 12 or not re.search(r"[a-zA-Z]{3}", block)
                or re.fullmatch(r"#+[^\n]*", block) or block.startswith("<!--")):
            continue
        # Prefer explanatory prose over headings and link-only paragraphs.
        prose = len(block.split()) >= 5 and bool(re.search(r"[.!?](?:\s|$)", block))
        priority = 5 if prose else 2
        if block.startswith(("* ", "- ")):
            priority = 4
        if len(block) <= MAX_QUOTE:
            candidates.append((priority, block, False))
        else:
            # Retain a spread fallback for directory listings and unstructured text.
            pending = ""
            for line in block.splitlines(keepends=True):
                if len(pending) + len(line) > MAX_QUOTE and pending.strip():
                    candidates.append((priority, pending.strip(), True))
                    pending = ""
                while len(line) > MAX_QUOTE:
                    quote = _prefix(line)
                    candidates.append((priority, quote, True))
                    line = line[len(quote):]
                pending += line
            if pending.strip():
                candidates.append((priority, pending.strip(), True))
    return candidates


def prepare_evidence(snapshot, max_excerpts=5):
    excerpts, sources = {}, []
    for source in snapshot["evidence"]:
        candidates = []
        seen = set()
        for priority, quote, partial in _candidates(source["text"], source["path"]):
            if 12 <= len(quote) <= MAX_QUOTE and quote not in seen:
                seen.add(quote)
                candidates.append((priority, quote, partial))
        selected = []
        # Prefer semantic blocks, with a spread across equally ranked candidates.
        for priority in sorted({c[0] for c in candidates}, reverse=True):
            group = [c for c in candidates if c[0] == priority
                     and not any(c[1] in s[1] or s[1] in c[1] for s in selected)]
            count = min(max(0, max_excerpts - len(selected)), len(group))
            indexes = ([0] if count == 1 else
                       sorted({round(i * (len(group) - 1) / (count - 1)) for i in range(count)})
                       if count else [])
            selected.extend(group[i] for i in indexes)
        available = []
        for i, (_, quote, partial) in enumerate(selected, 1):
            excerpt_id = f"excerpt-{source['id']}-{i:04d}"
            excerpts[excerpt_id] = {"evidence_id": source["id"], "quote": quote}
            item = {"excerpt_id": excerpt_id, "text": quote, "partial": partial}
            if partial or source["path"].endswith((".toml", ".yaml", ".yml", ".cfg")):
                # Keep the subject and qualifiers around a shortened quotation.
                # Context is evidence, not an extra citable quote or instruction.
                start = source["text"].find(quote)
                left = max(0, start - 300)
                left = source["text"].rfind("\n\n", left, start) + 2 if "\n\n" in source["text"][left:start] else left
                item["context"] = source["text"][left:min(len(source["text"]), start + len(quote) + 150)]
            available.append(item)
        sources.append({
            "id": source["id"], "path": source["path"],
            "scope": source_scope(source["path"]), "available_quotes": available,
            "selection_note": ("Selected excerpts only; partial excerpts may omit important context."
                               if available else "No usable excerpt; do not cite this source."),
            "truncated": bool(source.get("truncated", False) or
                              len(selected) < len(candidates) or
                              any(c[2] for c in selected) or
                              not selected),
        })
    return excerpts, sources
