from __future__ import annotations

from collections import Counter, defaultdict
import math

from src.data.vocab import tokenize


TokenList = list[str]
References = dict[str, list[str]]
Hypotheses = dict[str, str]


def _ngrams(tokens: TokenList, n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1))


def _closest_ref_len(hyp_len: int, ref_lens: list[int]) -> int:
    return min(ref_lens, key=lambda ref_len: (abs(ref_len - hyp_len), ref_len))


def _corpus_bleu(
    hypotheses: dict[str, TokenList],
    references: dict[str, list[TokenList]],
    max_n: int,
) -> float:
    clipped = [0 for _ in range(max_n)]
    total = [0 for _ in range(max_n)]
    hyp_len_sum = 0
    ref_len_sum = 0

    for image_name, hyp_tokens in hypotheses.items():
        ref_tokens_list = references[image_name]
        hyp_len_sum += len(hyp_tokens)
        ref_len_sum += _closest_ref_len(len(hyp_tokens), [len(ref) for ref in ref_tokens_list])

        for n in range(1, max_n + 1):
            hyp_counts = _ngrams(hyp_tokens, n)
            total[n - 1] += sum(hyp_counts.values())
            max_ref_counts: Counter[tuple[str, ...]] = Counter()
            for ref_tokens in ref_tokens_list:
                ref_counts = _ngrams(ref_tokens, n)
                for gram, count in ref_counts.items():
                    max_ref_counts[gram] = max(max_ref_counts[gram], count)
            clipped[n - 1] += sum(min(count, max_ref_counts[gram]) for gram, count in hyp_counts.items())

    if hyp_len_sum == 0:
        return 0.0
    brevity_penalty = 1.0 if hyp_len_sum > ref_len_sum else math.exp(1.0 - ref_len_sum / hyp_len_sum)
    precisions = []
    for i in range(max_n):
        if total[i] == 0:
            return 0.0
        precisions.append(max(clipped[i] / total[i], 1e-9))
    return brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / max_n)


def _lcs_len(a: TokenList, b: TokenList) -> int:
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def _rouge_l_sentence(hyp: TokenList, refs: list[TokenList]) -> float:
    best = 0.0
    for ref in refs:
        if not hyp or not ref:
            continue
        lcs = _lcs_len(hyp, ref)
        precision = lcs / len(hyp)
        recall = lcs / len(ref)
        if precision + recall == 0:
            score = 0.0
        else:
            score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def _meteor_match_positions(hyp: TokenList, ref: TokenList) -> list[tuple[int, int]]:
    used_ref: set[int] = set()
    matches: list[tuple[int, int]] = []
    for i, token in enumerate(hyp):
        for j, ref_token in enumerate(ref):
            if j not in used_ref and token == ref_token:
                matches.append((i, j))
                used_ref.add(j)
                break
    return matches


def _meteor_sentence(hyp: TokenList, refs: list[TokenList]) -> float:
    """精确词匹配版 METEOR，适合作为课程实验中的轻量实现。"""

    best = 0.0
    for ref in refs:
        matches = _meteor_match_positions(hyp, ref)
        match_count = len(matches)
        if match_count == 0 or not hyp or not ref:
            continue
        precision = match_count / len(hyp)
        recall = match_count / len(ref)
        f_mean = (10 * precision * recall) / (recall + 9 * precision)

        chunks = 1
        for idx in range(1, len(matches)):
            prev_h, prev_r = matches[idx - 1]
            cur_h, cur_r = matches[idx]
            if cur_h != prev_h + 1 or cur_r != prev_r + 1:
                chunks += 1
        penalty = 0.5 * (chunks / match_count) ** 3
        best = max(best, f_mean * (1 - penalty))
    return best


def _document_frequency(references: dict[str, list[TokenList]], n: int) -> Counter[tuple[str, ...]]:
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


def _cosine(a: dict[tuple[str, ...], float], b: dict[tuple[str, ...], float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(key, 0.0) for key, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _cider(
    hypotheses: dict[str, TokenList],
    references: dict[str, list[TokenList]],
    max_n: int = 4,
) -> float:
    num_docs = len(references)
    dfs = {n: _document_frequency(references, n) for n in range(1, max_n + 1)}
    image_scores: list[float] = []

    for image_name, hyp_tokens in hypotheses.items():
        refs = references[image_name]
        n_scores: list[float] = []
        for n in range(1, max_n + 1):
            hyp_vec = _tfidf_vector(hyp_tokens, n, dfs[n], num_docs)
            ref_sims = []
            for ref_tokens in refs:
                ref_vec = _tfidf_vector(ref_tokens, n, dfs[n], num_docs)
                ref_sims.append(_cosine(hyp_vec, ref_vec))
            n_scores.append(sum(ref_sims) / max(1, len(ref_sims)))
        image_scores.append(10.0 * sum(n_scores) / max_n)

    return sum(image_scores) / max(1, len(image_scores))


def compute_caption_metrics(
    hypotheses: Hypotheses,
    references: References,
) -> dict[str, float]:
    """计算图像描述常用指标。输入按 image_name 对齐。"""

    common_images = sorted(set(hypotheses) & set(references))
    if not common_images:
        raise ValueError("hypotheses 与 references 没有可对齐的图片")

    hyp_tokens = {name: tokenize(hypotheses[name]) for name in common_images}
    ref_tokens = {
        name: [tokenize(ref) for ref in references[name]]
        for name in common_images
    }

    rouge_scores = [_rouge_l_sentence(hyp_tokens[name], ref_tokens[name]) for name in common_images]
    meteor_scores = [_meteor_sentence(hyp_tokens[name], ref_tokens[name]) for name in common_images]

    return {
        "BLEU-1": _corpus_bleu(hyp_tokens, ref_tokens, max_n=1),
        "BLEU-2": _corpus_bleu(hyp_tokens, ref_tokens, max_n=2),
        "BLEU-3": _corpus_bleu(hyp_tokens, ref_tokens, max_n=3),
        "BLEU-4": _corpus_bleu(hyp_tokens, ref_tokens, max_n=4),
        "METEOR": sum(meteor_scores) / len(meteor_scores),
        "ROUGE-L": sum(rouge_scores) / len(rouge_scores),
        "CIDEr": _cider(hyp_tokens, ref_tokens, max_n=4),
    }
