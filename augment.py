"""Augmentation (A): turn retrieved chunks into the final user message.

Pipeline: gate on score -> dedupe -> budget -> position-aware order -> package.
Everything here is deterministic; no model is called.
"""
import re
from dataclasses import dataclass
from datetime import date


@dataclass
class Chunk:
    id: str
    title: str
    updated: date
    text: str
    score: float  # reranker relevance score, higher = better


@dataclass
class Augmented:
    user: str                 # the assembled message: <sources> + <question>
    sources: dict[str, str]   # id -> original chunk text (used later for validation)
    dropped: list[str]        # ids removed, for debugging/logging


def approx_tokens(text: str) -> int:
    """Rough estimate (~4 chars/token). Swap in your model's real tokenizer."""
    return max(1, len(text) // 4)


def _shingles(text: str, n: int = 5) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


def dedupe(chunks: list[Chunk], threshold: float = 0.8) -> list[Chunk]:
    """Drop near-duplicates (Jaccard similarity on word shingles), keeping the higher-scored one."""
    kept: list[Chunk] = []
    kept_shingles: list[set[str]] = []
    for c in sorted(chunks, key=lambda c: -c.score):
        s = _shingles(c.text)
        if all(len(s & k) / max(1, len(s | k)) < threshold for k in kept_shingles):
            kept.append(c)
            kept_shingles.append(s)
    return kept  # best-first


def select(chunks: list[Chunk], min_score: float = 0.35,
           token_budget: int = 3000, max_chunks: int = 10) -> list[Chunk]:
    """Relevance cutoff first, then a token budget. Cutoff beats a fixed top-k."""
    candidates = dedupe([c for c in chunks if c.score >= min_score])
    picked, used = [], 0
    for c in candidates:
        cost = approx_tokens(c.text)
        if len(picked) >= max_chunks:
            break
        if used + cost > token_budget:
            continue  # a smaller, lower-ranked chunk may still fit
        picked.append(c)
        used += cost
    return picked


def order_for_position(ranked: list[Chunk]) -> list[Chunk]:
    """Put the strongest evidence at the edges of the context, weakest in the middle.

    ranked = [1st, 2nd, 3rd, 4th, 5th] -> [1st, 3rd, 5th, 4th, 2nd]
    Trade-off: this breaks original document order. If your chunks are consecutive
    passages of one document, keep them together and in order instead.
    """
    front, back = [], []
    for i, c in enumerate(ranked):
        (front if i % 2 == 0 else back).append(c)
    return front + back[::-1]


def _package(chunk: Chunk) -> str:
    # Neutralise only the closing tag so a source can't break out of its wrapper.
    safe = chunk.text.replace("</source", "&lt;/source")
    return (f'<source id="{chunk.id}" title="{chunk.title}" updated="{chunk.updated.isoformat()}">\n'
            f"{safe}\n</source>")


def augment(question: str, retrieved: list[Chunk], **select_kwargs) -> Augmented | None:
    """Return the final user message, or None if retrieval was too weak to answer.

    None is the 'gate': the caller should reply "I couldn't find this" and skip generation.
    """
    picked = select(retrieved, **select_kwargs)
    if not picked:
        return None
    ordered = order_for_position(picked)
    body = "\n".join(_package(c) for c in ordered)
    user = f"<sources>\n{body}\n</sources>\n\n<question>{question}</question>"
    kept_ids = {c.id for c in picked}
    return Augmented(
        user=user,
        sources={c.id: c.text for c in picked},
        dropped=[c.id for c in retrieved if c.id not in kept_ids],
    )
