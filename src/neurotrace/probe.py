"""Activation geometry probe: find the sabotage direction in activation space."""

import json
import os
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ProbeResult:
    """Results from a linear probe analysis."""

    dataset_name: str
    model_name: str
    layer: int
    extraction_point: str  # "pre_mlp", "post_attn", "post_mlp"
    num_clean: int
    num_sabotaged: int
    activations: np.ndarray  # (num_prompts, hidden_dim)
    labels: np.ndarray  # boolean, True = sabotaged
    prompts: list[str]
    mean_direction: np.ndarray  # (hidden_dim,)
    probe_direction: np.ndarray  # (hidden_dim,)
    projection_scores: np.ndarray  # (num_prompts,)
    cohens_d: float
    auc_roc: float
    probe_accuracy: float
    probe_correct: int
    probe_total: int
    direction_alignment: float  # cosine sim between mean diff and probe
    pca_components: np.ndarray  # (num_prompts, 3)
    pca_explained_variance: np.ndarray  # (3,)
    cross_dataset: str | None = None
    cross_auc_roc: float | None = None
    cross_prompts: list[str] | None = None
    cross_labels: np.ndarray | None = None
    cross_scores: np.ndarray | None = None


def extract_activations(
    model,
    tokenizer,
    prompts: list[str],
    layer: int,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Extract residual stream activations at three points for a target layer.

    Returns dict with keys: "pre_mlp", "post_attn", "post_mlp", each (N, hidden_dim).
    """
    from neurotrace.models import get_architecture

    arch = get_architecture(model.config.model_type)
    layers = arch.get_layers(model)

    pre_mlp_acts = []
    post_attn_acts = []
    post_mlp_acts = []

    for prompt in prompts:
        torch.manual_seed(seed)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(next(model.parameters()).device)

        captured = {}

        # Hook on the layer itself to get input (pre_mlp approximation)
        # and output (post_mlp)
        def layer_hook(_module, input, output):
            # input[0] is the hidden states entering this layer
            captured["layer_input"] = input[0].detach()
            out_tensor = output[0] if isinstance(output, tuple) else output
            captured["layer_output"] = out_tensor.detach()

        # Hook on attention to get post-attention output
        def attn_hook(_module, _input, output):
            out_tensor = output[0] if isinstance(output, tuple) else output
            captured["attn_output"] = out_tensor.detach()

        target_layer = layers[layer]
        attn_module = arch.get_attention(target_layer)

        h_layer = target_layer.register_forward_hook(layer_hook)
        h_attn = attn_module.register_forward_hook(attn_hook)

        try:
            with torch.no_grad():
                model(input_ids)
        finally:
            h_layer.remove()
            h_attn.remove()

        # Last token position
        pre_mlp_acts.append(captured["layer_input"][0, -1, :].cpu().numpy())
        # post_attn: layer input + attention output
        # The residual stream after attention is: input + attn_output
        attn_out = captured["attn_output"][0, -1, :].cpu().numpy()
        layer_in = captured["layer_input"][0, -1, :].cpu().numpy()
        post_attn_acts.append(layer_in + attn_out)
        post_mlp_acts.append(captured["layer_output"][0, -1, :].cpu().numpy())

    return {
        "pre_mlp": np.stack(pre_mlp_acts),
        "post_attn": np.stack(post_attn_acts),
        "post_mlp": np.stack(post_mlp_acts),
    }


def label_from_scan(scan_result) -> tuple[list[str], np.ndarray, list[int]]:
    """Extract prompts and labels from scan results.

    Returns (prompts, labels, included_indices) where labels is boolean
    (True = sabotaged). Excludes 'wrong' and 'weak' statuses.
    """
    prompts = []
    labels = []
    indices = []
    for i, pr in enumerate(scan_result.prompt_results):
        if pr.status in ("correct", "sabotaged"):
            prompts.append(pr.prompt)
            labels.append(pr.status == "sabotaged")
            indices.append(i)
    return prompts, np.array(labels, dtype=bool), indices


def compute_mean_direction(
    activations: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Compute the normalized mean-difference sabotage direction."""
    mean_sabo = activations[labels].mean(axis=0)
    mean_clean = activations[~labels].mean(axis=0)
    direction = mean_sabo - mean_clean
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    return direction


def train_probe(
    activations: np.ndarray, labels: np.ndarray
) -> tuple[float, int, int, np.ndarray]:
    """Train logistic regression with LOO cross-validation.

    Returns (accuracy, correct_count, total, probe_direction).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut

    loo = LeaveOneOut()
    correct = 0
    for train_idx, test_idx in loo.split(activations):
        probe = LogisticRegression(max_iter=1000)
        probe.fit(activations[train_idx], labels[train_idx])
        if probe.predict(activations[test_idx]) == labels[test_idx]:
            correct += 1
    total = len(labels)
    accuracy = correct / total

    # Train final probe on all data for the direction vector
    final_probe = LogisticRegression(max_iter=1000)
    final_probe.fit(activations, labels)
    probe_direction = final_probe.coef_[0]
    norm = np.linalg.norm(probe_direction)
    if norm > 0:
        probe_direction = probe_direction / norm

    return accuracy, correct, total, probe_direction


def compute_projection_scores(
    activations: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Project activations onto a direction vector."""
    return activations @ direction


def compute_cohens_d(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute Cohen's d effect size between sabotaged and clean groups."""
    sabo_scores = scores[labels]
    clean_scores = scores[~labels]
    if len(sabo_scores) < 2 or len(clean_scores) < 2:
        return 0.0
    mean_diff = sabo_scores.mean() - clean_scores.mean()
    n1, n2 = len(sabo_scores), len(clean_scores)
    s1 = (n1 - 1) * sabo_scores.var(ddof=1)
    s2 = (n2 - 1) * clean_scores.var(ddof=1)
    pooled_var = (s1 + s2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)
    if pooled_std == 0:
        return 0.0
    return float(mean_diff / pooled_std)


def compute_auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUC-ROC using projection scores as classifier."""
    from sklearn.metrics import roc_auc_score

    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5  # degenerate case
    return float(roc_auc_score(labels.astype(int), scores))


def run_pca(
    activations: np.ndarray, n_components: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Run PCA and return (projected, explained_variance_ratio)."""
    from sklearn.decomposition import PCA

    n_components = min(n_components, activations.shape[0], activations.shape[1])
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(activations)
    return projected, pca.explained_variance_ratio_


def auto_detect_layer(scan_result) -> int | None:
    """Auto-detect the primary sabotage layer from scan results."""
    from collections import Counter

    layer_counts: Counter = Counter()
    for pr in scan_result.prompt_results:
        if pr.status == "sabotaged":
            for layer in pr.sabotage_layers:
                layer_counts[layer] += 1
    if not layer_counts:
        return None
    return layer_counts.most_common(1)[0][0]


def run_probe(
    model,
    tokenizer,
    scan_result,
    layer: int,
    extraction_point: str = "pre_mlp",
    seed: int = 42,
    cross_scan_result=None,
    cross_dataset_name: str | None = None,
    progress_callback=None,
) -> ProbeResult:
    """Run the full probe analysis pipeline.

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        scan_result: ScanResult from running scan
        layer: Layer to extract activations from
        extraction_point: Which activation to analyze
        seed: Random seed
        cross_scan_result: Optional ScanResult for cross-domain test
        cross_dataset_name: Name of cross dataset
        progress_callback: Optional callable(description)
    """
    model_name = model.config._name_or_path

    # Step 1-2: Label and filter
    if progress_callback:
        progress_callback("Labeling prompts...")
    prompts, labels, _indices = label_from_scan(scan_result)

    if labels.sum() < 1:
        raise ValueError("No sabotaged prompts found in scan results.")

    # Step 1: Extract activations
    if progress_callback:
        progress_callback(f"Extracting activations at layer {layer}...")
    all_acts = extract_activations(model, tokenizer, prompts, layer, seed=seed)
    activations = all_acts[extraction_point]

    # Step 3: Mean difference direction
    if progress_callback:
        progress_callback("Computing mean difference direction...")
    mean_dir = compute_mean_direction(activations, labels)

    # Step 4: Linear probe
    if progress_callback:
        progress_callback("Training linear probe (LOO)...")
    accuracy, correct, total, probe_dir = train_probe(activations, labels)

    # Step 5: Projection scores
    scores = compute_projection_scores(activations, mean_dir)
    cohens_d = compute_cohens_d(scores, labels)
    auc_roc = compute_auc_roc(scores, labels)

    # Direction alignment
    direction_alignment = float(np.dot(mean_dir, probe_dir))

    # Step 6: PCA
    if progress_callback:
        progress_callback("Running PCA...")
    pca_proj, pca_var = run_pca(activations)

    # Step 7: Cross-domain
    cross_auc = None
    cross_prompts_out = None
    cross_labels_out = None
    cross_scores_out = None
    if cross_scan_result is not None:
        if progress_callback:
            progress_callback("Cross-domain projection...")
        cross_prompts_list, cross_labels_arr, _ = label_from_scan(cross_scan_result)
        if len(cross_prompts_list) > 0 and cross_labels_arr.sum() > 0:
            cross_acts = extract_activations(
                model, tokenizer, cross_prompts_list, layer, seed=seed
            )
            cross_activations = cross_acts[extraction_point]
            cross_scores_arr = compute_projection_scores(cross_activations, mean_dir)
            cross_auc = compute_auc_roc(cross_scores_arr, cross_labels_arr)
            cross_prompts_out = cross_prompts_list
            cross_labels_out = cross_labels_arr
            cross_scores_out = cross_scores_arr

    return ProbeResult(
        dataset_name=scan_result.dataset_name,
        model_name=model_name,
        layer=layer,
        extraction_point=extraction_point,
        num_clean=int((~labels).sum()),
        num_sabotaged=int(labels.sum()),
        activations=activations,
        labels=labels,
        prompts=prompts,
        mean_direction=mean_dir,
        probe_direction=probe_dir,
        projection_scores=scores,
        cohens_d=cohens_d,
        auc_roc=auc_roc,
        probe_accuracy=accuracy,
        probe_correct=correct,
        probe_total=total,
        direction_alignment=direction_alignment,
        pca_components=pca_proj,
        pca_explained_variance=pca_var,
        cross_dataset=cross_dataset_name,
        cross_auc_roc=cross_auc,
        cross_prompts=cross_prompts_out,
        cross_labels=cross_labels_out,
        cross_scores=cross_scores_out,
    )


def save_probe_outputs(result: ProbeResult, output_dir: str) -> None:
    """Save all probe outputs to the output directory."""
    os.makedirs(output_dir, exist_ok=True)

    # activations.npz
    np.savez(
        os.path.join(output_dir, "activations.npz"),
        activations=result.activations,
        labels=result.labels,
    )

    # mean_direction.npy
    np.save(os.path.join(output_dir, "mean_direction.npy"), result.mean_direction)

    # probe_direction.npy
    np.save(os.path.join(output_dir, "probe_direction.npy"), result.probe_direction)

    # projection_scores.json
    scores_data = []
    for i, prompt in enumerate(result.prompts):
        scores_data.append({
            "prompt": prompt,
            "label": "sabotaged" if result.labels[i] else "clean",
            "score": float(result.projection_scores[i]),
        })
    with open(os.path.join(output_dir, "projection_scores.json"), "w") as f:
        json.dump(scores_data, f, indent=2)

    # pca_data.json
    pca_data = {
        "pca_components": result.pca_components.tolist(),
        "labels": ["sabotaged" if lb else "clean" for lb in result.labels],
        "prompts": result.prompts,
        "explained_variance": result.pca_explained_variance.tolist(),
    }
    with open(os.path.join(output_dir, "pca_data.json"), "w") as f:
        json.dump(pca_data, f, indent=2)

    # cross_domain.json (if applicable)
    if result.cross_dataset is not None and result.cross_scores is not None:
        cross_data = {
            "train_dataset": result.dataset_name,
            "test_dataset": result.cross_dataset,
            "auc_roc": result.cross_auc_roc,
            "prompts": result.cross_prompts,
            "labels": ["sabotaged" if lb else "clean"
                for lb in (
                    result.cross_labels
                    if result.cross_labels is not None
                    else []
                )],
            "scores": result.cross_scores.tolist(),
        }
        with open(os.path.join(output_dir, "cross_domain.json"), "w") as f:
            json.dump(cross_data, f, indent=2)

    # report.md
    report = generate_report(result)
    with open(os.path.join(output_dir, "report.md"), "w") as f:
        f.write(report)


def generate_report(result: ProbeResult) -> str:
    """Generate a human-readable markdown report."""
    total_in_dataset = result.num_clean + result.num_sabotaged

    lines = [
        f"# Probe Analysis: {result.dataset_name}",
        "",
        "## Dataset",
        f"- Total prompts analyzed: {total_in_dataset}",
        f"- Sabotaged: {result.num_sabotaged}",
        f"- Clean: {result.num_clean}",
        f"- Extraction point: {result.extraction_point}",
        f"- Layer: {result.layer}",
        "",
    ]

    if result.num_sabotaged < 5:
        n = result.num_sabotaged
        lines.extend([
            f"> **Warning:** Only {n} sabotaged examples."
            " Results may be unreliable.",
            "",
        ])

    # Mean difference analysis
    clean_scores = result.projection_scores[~result.labels]
    sabo_scores = result.projection_scores[result.labels]

    align = result.direction_alignment
    c_mean = f"{clean_scores.mean():.4f} +/- {clean_scores.std():.4f}"
    s_mean = f"{sabo_scores.mean():.4f} +/- {sabo_scores.std():.4f}"
    lines.extend([
        "## Mean Difference Analysis",
        f"- Sabotage direction cosine sim with probe: {align:.4f}",
        f"- Mean projection (clean): {c_mean}",
        f"- Mean projection (sabotaged): {s_mean}",
        f"- Cohen's d: {result.cohens_d:.4f}",
        f"- AUC-ROC: {result.auc_roc:.4f}",
        "",
    ])

    # Linear probe
    acc = f"{result.probe_correct}/{result.probe_total}"
    lines.extend([
        "## Linear Probe (Leave-One-Out)",
        f"- Accuracy: {acc} ({result.probe_accuracy:.1%})",
        f"- Direction alignment with mean diff:"
        f" {result.direction_alignment:.4f} (cosine sim)",
        "",
    ])

    # Per-prompt projections
    lines.extend([
        "## Per-Prompt Projections",
        "| Prompt | Label | Score | Predicted |",
        "|--------|-------|-------|-----------|",
    ])
    threshold = (clean_scores.mean() + sabo_scores.mean()) / 2
    for i, prompt in enumerate(result.prompts):
        label = "sabotaged" if result.labels[i] else "clean"
        score = result.projection_scores[i]
        predicted = "sabotaged" if score > threshold else "clean"
        prompt_display = prompt if len(prompt) <= 40 else prompt[:37] + "..."
        lines.append(f"| {prompt_display} | {label} | {score:.4f} | {predicted} |")
    lines.append("")

    # PCA summary
    pca_var = result.pca_explained_variance
    lines.extend([
        "## PCA Summary",
        f"- Variance explained: PC1={pca_var[0]:.1%},"
        f" PC2={pca_var[1]:.1%}, PC3={pca_var[2]:.1%}",
        "",
    ])

    # Cross-domain
    if result.cross_dataset is not None:
        lines.extend([
            "## Cross-Domain",
            f"- Direction trained on: {result.dataset_name}",
            f"- Tested on: {result.cross_dataset}",
            f"- AUC-ROC on cross dataset: {result.cross_auc_roc:.4f}",
            "",
        ])

    return "\n".join(lines)
