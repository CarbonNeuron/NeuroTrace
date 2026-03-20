"""Universal vulnerability probe: predict MLP ablation vulnerability across domains."""

import json
import os
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class UniversalProbeResult:
    """Results from a universal vulnerability probe."""

    heatmap_run_ids: list[str]
    model_name: str
    layer_range: tuple[int, int]  # (start, end) inclusive
    n_samples: int
    n_vulnerable: int
    n_robust: int
    n_excluded: int
    auc_roc: float
    cohens_d: float
    accuracy: float
    per_domain_auc: dict[str, float]
    confusion_matrix: dict[str, int]  # tp, fp, tn, fn
    top_features: list[dict]  # [{layer, importance}, ...]
    prompts: list[str]
    labels: np.ndarray  # boolean, True = vulnerable
    domains: list[str]  # domain name per sample
    probe_direction: np.ndarray  # (feature_dim,)
    projection_scores: np.ndarray  # (n_samples,)


def label_from_heatmap(cells_json: str) -> dict[int, dict]:
    """Extract per-prompt vulnerability labels from heatmap cells JSON.

    Returns dict keyed by prompt_index with:
        prompt, expected_answer, baseline_correct, has_break
    """
    cells = json.loads(cells_json)
    prompts: dict[int, dict] = {}
    for c in cells:
        idx = c["prompt_index"]
        if idx not in prompts:
            prompts[idx] = {
                "prompt": c["prompt"],
                "expected_answer": c["expected_answer"],
                "baseline_correct": c["baseline_correct"],
                "has_break": False,
            }
        if c["flip_direction"] == "broke":
            prompts[idx]["has_break"] = True
    return prompts


def build_labels_from_heatmap_runs(
    heatmap_runs: list[dict],
) -> tuple[list[str], np.ndarray, list[str]]:
    """Build prompts, labels, and domain names from multiple heatmap runs.

    Returns (prompts, labels, domains) excluding baseline-wrong facts.
    """
    all_prompts: list[str] = []
    all_labels: list[bool] = []
    all_domains: list[str] = []

    for run in heatmap_runs:
        cells_json = run["cells"]
        dataset_name = run["dataset_name"]
        prompt_info = label_from_heatmap(cells_json)

        for _idx, info in sorted(prompt_info.items()):
            if not info["baseline_correct"]:
                continue  # exclude baseline-wrong
            all_prompts.append(info["prompt"])
            all_labels.append(info["has_break"])
            all_domains.append(dataset_name)

    return all_prompts, np.array(all_labels, dtype=bool), all_domains


def extract_multilayer_activations(
    model,
    tokenizer,
    prompts: list[str],
    layer_start: int,
    layer_end: int,
    seed: int = 42,
) -> np.ndarray:
    """Extract and concatenate MLP output activations from a range of layers.

    Returns array of shape (n_prompts, (layer_end - layer_start + 1) * hidden_dim).
    """
    from neurotrace.models import get_architecture

    arch = get_architecture(model.config.model_type)
    layers = arch.get_layers(model)

    all_acts = []
    for prompt in prompts:
        torch.manual_seed(seed)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(next(model.parameters()).device)

        captured: dict[int, torch.Tensor] = {}

        hooks = []
        for layer_idx in range(layer_start, layer_end + 1):
            target_layer = layers[layer_idx]

            def make_hook(li):
                def hook_fn(_module, _input, output):
                    out_tensor = output[0] if isinstance(output, tuple) else output
                    captured[li] = out_tensor.detach()
                return hook_fn

            h = target_layer.register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

        try:
            with torch.no_grad():
                model(input_ids)
        finally:
            for h in hooks:
                h.remove()

        layer_acts = []
        for layer_idx in range(layer_start, layer_end + 1):
            # Last token position, full hidden dim
            act = captured[layer_idx][0, -1, :].cpu().numpy()
            layer_acts.append(act)

        all_acts.append(np.concatenate(layer_acts))

    return np.stack(all_acts)


def train_universal_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    domains: list[str],
) -> UniversalProbeResult:
    """Train a logistic regression probe with LOO CV.

    Returns a populated UniversalProbeResult (caller fills
    heatmap_run_ids, model_name, layer_range).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(activations)

    # LOO cross-validation
    loo = LeaveOneOut()
    predictions = np.zeros(len(labels), dtype=bool)
    decision_scores = np.zeros(len(labels), dtype=np.float64)

    for train_idx, test_idx in loo.split(x_scaled):
        probe = LogisticRegression(max_iter=2000)
        probe.fit(x_scaled[train_idx], labels[train_idx])
        predictions[test_idx] = probe.predict(x_scaled[test_idx])
        decision_scores[test_idx] = probe.decision_function(
            x_scaled[test_idx]
        )

    correct = (predictions == labels).sum()
    accuracy = float(correct / len(labels))

    # AUC-ROC
    if labels.sum() == 0 or labels.sum() == len(labels):
        auc_roc = 0.5
    else:
        auc_roc = float(roc_auc_score(labels.astype(int), decision_scores))

    # Cohen's d
    vuln_scores = decision_scores[labels]
    robust_scores = decision_scores[~labels]
    cohens_d = _cohens_d(vuln_scores, robust_scores)

    # Confusion matrix
    cm = sk_confusion_matrix(labels, predictions, labels=[False, True])
    tn, fp, fn, tp = cm.ravel()
    cm_dict = {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}

    # Per-domain AUC
    unique_domains = sorted(set(domains))
    per_domain_auc: dict[str, float] = {}
    for domain in unique_domains:
        mask = np.array([d == domain for d in domains])
        domain_labels = labels[mask]
        domain_scores = decision_scores[mask]
        if domain_labels.sum() == 0 or domain_labels.sum() == len(domain_labels):
            per_domain_auc[domain] = 0.5
        else:
            per_domain_auc[domain] = float(
                roc_auc_score(domain_labels.astype(int), domain_scores)
            )

    # Train final probe on all data for direction + feature importance
    final_probe = LogisticRegression(max_iter=2000)
    final_probe.fit(x_scaled, labels)
    probe_direction = final_probe.coef_[0]

    top_features = _compute_feature_importance(probe_direction)

    # Projection scores using final probe direction
    proj_dir = probe_direction / (np.linalg.norm(probe_direction) or 1.0)
    projection_scores = x_scaled @ proj_dir

    return UniversalProbeResult(
        heatmap_run_ids=[],
        model_name="",
        layer_range=(0, 0),
        n_samples=len(labels),
        n_vulnerable=int(labels.sum()),
        n_robust=int((~labels).sum()),
        n_excluded=0,
        auc_roc=auc_roc,
        cohens_d=cohens_d,
        accuracy=accuracy,
        per_domain_auc=per_domain_auc,
        confusion_matrix=cm_dict,
        top_features=top_features,
        prompts=[],
        labels=labels,
        domains=domains,
        probe_direction=proj_dir,
        projection_scores=projection_scores,
    )


def _cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Cohen's d between two groups."""
    if len(group_a) < 2 or len(group_b) < 2:
        return 0.0
    mean_diff = group_a.mean() - group_b.mean()
    n1, n2 = len(group_a), len(group_b)
    s1 = (n1 - 1) * group_a.var(ddof=1)
    s2 = (n2 - 1) * group_b.var(ddof=1)
    pooled_var = (s1 + s2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)
    if pooled_std == 0:
        return 0.0
    return float(mean_diff / pooled_std)


def _compute_feature_importance(
    coef: np.ndarray,
) -> list[dict]:
    """Compute per-feature absolute importance from probe coefficients.

    Returns top 20 features sorted by importance.
    """
    importance = np.abs(coef)
    top_indices = np.argsort(importance)[::-1][:20]
    return [
        {"feature_index": int(idx), "importance": float(importance[idx])}
        for idx in top_indices
    ]


def save_universal_probe(
    result: UniversalProbeResult,
    output_dir: str,
    scaler=None,
    probe_model=None,
) -> None:
    """Save probe outputs to the output directory."""
    os.makedirs(output_dir, exist_ok=True)

    # Save probe direction
    np.save(
        os.path.join(output_dir, "probe_direction.npy"),
        result.probe_direction,
    )

    # Save the sklearn model if provided
    if probe_model is not None:
        import joblib

        joblib.dump(probe_model, os.path.join(output_dir, "probe_model.joblib"))
    if scaler is not None:
        import joblib

        joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))

    # Save summary JSON
    summary = {
        "heatmap_run_ids": result.heatmap_run_ids,
        "model_name": result.model_name,
        "layer_range": list(result.layer_range),
        "n_samples": result.n_samples,
        "n_vulnerable": result.n_vulnerable,
        "n_robust": result.n_robust,
        "n_excluded": result.n_excluded,
        "auc_roc": result.auc_roc,
        "cohens_d": result.cohens_d,
        "accuracy": result.accuracy,
        "per_domain_auc": result.per_domain_auc,
        "confusion_matrix": result.confusion_matrix,
        "top_features": result.top_features,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Per-prompt scores
    scores_data = []
    for i, prompt in enumerate(result.prompts):
        scores_data.append({
            "prompt": prompt,
            "domain": result.domains[i],
            "label": "vulnerable" if result.labels[i] else "robust",
            "score": float(result.projection_scores[i]),
        })
    with open(os.path.join(output_dir, "projection_scores.json"), "w") as f:
        json.dump(scores_data, f, indent=2)
