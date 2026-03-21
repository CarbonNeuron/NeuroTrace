"""Chain trace — multi-hop reasoning through residual streams."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChainResult:
    """Result of a chain trace for one prompt."""

    prompt: str
    intermediate: str
    answer: str
    intermediate_logits: list[float]  # per-layer cumulative logit for intermediate
    answer_logits: list[float]  # per-layer cumulative logit for answer
    intermediate_ranks: list[int]  # per-layer rank of intermediate token
    answer_ranks: list[int]  # per-layer rank of answer token
    intermediate_commit_layer: int | None  # first layer where intermediate enters top-10
    answer_commit_layer: int | None  # first layer where answer enters top-10
    handoff_gap: int | None  # answer_commit - intermediate_commit
    intermediate_peak_rank: int  # best rank intermediate achieves
    answer_peak_rank: int  # best rank answer achieves
    chain_status: str  # COMPLETE, PARTIAL, BROKEN, SHORTCUT
    num_layers: int


def classify_chain(
    answer_rank_final: int,
    intermediate_peak_rank: int,
) -> str:
    """Classify chain status based on final answer rank and intermediate peak rank."""
    if answer_rank_final == 1:
        if intermediate_peak_rank <= 10:
            return "COMPLETE"
        else:
            return "SHORTCUT"
    elif intermediate_peak_rank <= 10:
        return "PARTIAL"
    else:
        return "BROKEN"


def _first_top_k_layer(ranks: list[int], k: int = 10) -> int | None:
    """Return first layer index where rank <= k, or None."""
    for i, r in enumerate(ranks):
        if r <= k:
            return i
    return None


def run_chain_remote(
    client,
    prompt: str,
    intermediate: str,
    answer: str,
    *,
    raw: bool = True,
    seed: int = 42,
) -> ChainResult:
    """Run chain trace via remote worker using decompose endpoint."""
    int_result = client.decompose(prompt, intermediate, raw=raw, seed=seed)
    ans_result = client.decompose(prompt, answer, raw=raw, seed=seed)

    num_layers = len(int_result.layers)

    int_cumulatives = [lc.cumulative for lc in int_result.layers]
    ans_cumulatives = [lc.cumulative for lc in ans_result.layers]

    # Use cumulative_rank from decompose response (computed from full vocab
    # logits at each cumulative residual state). The forward endpoint's
    # layer_predictions are delta-based, not cumulative, so ranks from those
    # are incorrect.
    int_ranks = [
        lc.cumulative_rank if lc.cumulative_rank is not None else 999
        for lc in int_result.layers
    ]
    ans_ranks = [
        lc.cumulative_rank if lc.cumulative_rank is not None else 999
        for lc in ans_result.layers
    ]

    int_peak = min(int_ranks) if int_ranks else 999
    ans_peak = min(ans_ranks) if ans_ranks else 999
    ans_final = ans_ranks[-1] if ans_ranks else 999

    int_commit = _first_top_k_layer(int_ranks)
    ans_commit = _first_top_k_layer(ans_ranks)

    if int_commit is not None and ans_commit is not None:
        gap = ans_commit - int_commit
    else:
        gap = None

    return ChainResult(
        prompt=prompt,
        intermediate=intermediate,
        answer=answer,
        intermediate_logits=int_cumulatives,
        answer_logits=ans_cumulatives,
        intermediate_ranks=int_ranks,
        answer_ranks=ans_ranks,
        intermediate_commit_layer=int_commit,
        answer_commit_layer=ans_commit,
        handoff_gap=gap,
        intermediate_peak_rank=int_peak,
        answer_peak_rank=ans_peak,
        chain_status=classify_chain(ans_final, int_peak),
        num_layers=num_layers,
    )


def run_chain_local(
    model,
    tokenizer,
    prompt: str,
    intermediate: str,
    answer: str,
    *,
    seed: int = 42,
) -> ChainResult:
    """Run chain trace locally using decompose_logits."""
    from neurotrace.decompose import decompose_logits

    decomps = decompose_logits(
        model, tokenizer, prompt,
        [intermediate, answer],
        seed=seed,
    )

    int_decomp = decomps[intermediate]
    ans_decomp = decomps[answer]
    num_layers = len(int_decomp.layers)

    # Build cumulative logits
    int_cum = []
    ans_cum = []
    int_total = int_decomp.embedding
    ans_total = ans_decomp.embedding
    for il, al in zip(int_decomp.layers, ans_decomp.layers):
        int_total += il.attention + il.mlp
        ans_total += al.attention + al.mlp
        int_cum.append(int_total)
        ans_cum.append(ans_total)

    # Get per-layer ranks using layer predictions
    import torch

    torch.manual_seed(seed)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Resolve token IDs
    int_ids = tokenizer.encode(intermediate, add_special_tokens=False)
    if not int_ids:
        int_ids = tokenizer.encode(" " + intermediate, add_special_tokens=False)
    int_id = int_ids[0] if int_ids else 0

    ans_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not ans_ids:
        ans_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    ans_id = ans_ids[0] if ans_ids else 0

    # Capture residual stream at each layer
    residuals: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_idx: int):
        def hook(module, input, output):
            residuals[layer_idx] = output[0][0, -1, :].detach()
        return hook

    layers = model.model.layers
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    # Project residuals through final norm + lm_head to get per-layer ranks
    ln_weight = model.model.norm.weight.detach()
    eps = model.model.norm.variance_epsilon
    w_unembed = model.lm_head.weight

    int_ranks = []
    ans_ranks = []
    for i in range(num_layers):
        res = residuals[i]
        variance = res.pow(2).mean(-1, keepdim=False)
        s = torch.sqrt(variance + eps)
        normed = (res * ln_weight) / s
        logits = normed @ w_unembed.T
        sorted_ids = torch.argsort(logits, descending=True)
        rank_list = sorted_ids.tolist()
        int_rank = rank_list.index(int_id) + 1 if int_id in rank_list else 999
        ans_rank = rank_list.index(ans_id) + 1 if ans_id in rank_list else 999
        int_ranks.append(int_rank)
        ans_ranks.append(ans_rank)

    int_peak = min(int_ranks) if int_ranks else 999
    ans_peak = min(ans_ranks) if ans_ranks else 999
    ans_final = ans_ranks[-1] if ans_ranks else 999

    int_commit = _first_top_k_layer(int_ranks)
    ans_commit = _first_top_k_layer(ans_ranks)

    if int_commit is not None and ans_commit is not None:
        gap = ans_commit - int_commit
    else:
        gap = None

    return ChainResult(
        prompt=prompt,
        intermediate=intermediate,
        answer=answer,
        intermediate_logits=int_cum,
        answer_logits=ans_cum,
        intermediate_ranks=int_ranks,
        answer_ranks=ans_ranks,
        intermediate_commit_layer=int_commit,
        answer_commit_layer=ans_commit,
        handoff_gap=gap,
        intermediate_peak_rank=int_peak,
        answer_peak_rank=ans_peak,
        chain_status=classify_chain(ans_final, int_peak),
        num_layers=num_layers,
    )


def run_chain_batch(
    results_fn,
    entries: list[dict],
) -> list[ChainResult]:
    """Run chain trace for a batch of dataset entries.

    results_fn: callable(prompt, intermediate, answer) -> ChainResult
    """
    results = []
    for entry in entries:
        result = results_fn(
            entry["prompt"],
            entry["intermediate"],
            entry["answer"],
        )
        results.append(result)
    return results


def chain_result_to_dict(r: ChainResult) -> dict:
    """Serialize ChainResult to dict for JSON output."""
    return {
        "prompt": r.prompt,
        "intermediate": r.intermediate,
        "answer": r.answer,
        "intermediate_logits": r.intermediate_logits,
        "answer_logits": r.answer_logits,
        "intermediate_ranks": r.intermediate_ranks,
        "answer_ranks": r.answer_ranks,
        "intermediate_commit_layer": r.intermediate_commit_layer,
        "answer_commit_layer": r.answer_commit_layer,
        "handoff_gap": r.handoff_gap,
        "intermediate_peak_rank": r.intermediate_peak_rank,
        "answer_peak_rank": r.answer_peak_rank,
        "chain_status": r.chain_status,
        "num_layers": r.num_layers,
    }


def _find_rank(top_tokens, token_str: str) -> int:
    """Find rank of token_str in a list of TokenPrediction objects."""
    token_lower = token_str.lower().strip()
    for i, tp in enumerate(top_tokens):
        if tp.token.strip().lower() == token_lower:
            return i + 1
    return 999


def _pad_to(lst: list, length: int, fill) -> list:
    """Pad or trim list to exact length."""
    if len(lst) >= length:
        return lst[:length]
    return lst + [fill] * (length - len(lst))
