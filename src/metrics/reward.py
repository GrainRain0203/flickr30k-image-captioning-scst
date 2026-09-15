from __future__ import annotations

from collections import Counter
import math

from src.data.vocab import tokenize


TokenList = list[str]


def _ngrams(tokens: TokenList, n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1))


def _document_frequency(
    references: dict[str, list[TokenList]],
    n: int,
) -> Counter[tuple[str, ...]]:
    df: Counter[tuple[str, ...]] = Counter()
    for refs in references.values():
        grams_in_image: set[tuple[str, ...]] = set()
        for ref in refs:
            grams_in_image.update(_ngrams(ref, n))
        df.update(grams_in_image)
    return df


def _tfidf_vector(
    tokens: TokenList,
    n: int,
    df: Counter[tuple[str, ...]],
    num_docs: int,
) -> dict[tuple[str, ...], float]:
    counts = _ngrams(tokens, n)
    total = sum(counts.values())
    if total == 0:
        return {}

    vector: dict[tuple[str, ...], float] = {}
    for gram, count in counts.items():
        tf = count / total
        idf = math.log((num_docs + 1.0) / (df.get(gram, 0) + 1.0))
        vector[gram] = tf * idf
    return vector


def _cosine(
    a: dict[tuple[str, ...], float],
    b: dict[tuple[str, ...], float],
) -> float:
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(key, 0.0) for key, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class CiderReward:
    """Fast per-image CIDEr-style reward for SCST training."""

    def __init__(self, references_by_image: dict[str, list[str]], max_n: int = 4) -> None:
        self.max_n = max_n
        self.references = {
            image_name: [tokenize(ref) for ref in refs]
            for image_name, refs in references_by_image.items()
        }
        self.num_docs = len(self.references)
        if self.num_docs == 0:
            raise ValueError("CiderReward needs at least one reference image")
        self.dfs = {
            n: _document_frequency(self.references, n)
            for n in range(1, max_n + 1)
        }

    def score(self, image_name: str, hypothesis: str) -> float:
        if image_name not in self.references:
            raise KeyError(f"Missing references for image: {image_name}")

        hyp_tokens = tokenize(hypothesis)
        refs = self.references[image_name]
        n_scores: list[float] = []
        for n in range(1, self.max_n + 1):
            hyp_vec = _tfidf_vector(hyp_tokens, n, self.dfs[n], self.num_docs)
            ref_sims = [
                _cosine(
                    hyp_vec,
                    _tfidf_vector(ref_tokens, n, self.dfs[n], self.num_docs),
                )
                for ref_tokens in refs
            ]
            n_scores.append(sum(ref_sims) / max(1, len(ref_sims)))
        return 10.0 * sum(n_scores) / self.max_n

    def score_many(self, image_names: list[str], hypotheses: list[str]) -> list[float]:
        if len(image_names) != len(hypotheses):
            raise ValueError("image_names and hypotheses must have the same length")
        return [
            self.score(image_name, hypothesis)
            for image_name, hypothesis in zip(image_names, hypotheses)
        ]
