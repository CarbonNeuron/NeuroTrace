"""Tests for HTML report generation."""

import numpy as np
from click.testing import CliRunner

from neurotrace.cli import cli
from neurotrace.report import (
    _sample_layers,
    generate_comparison_report,
    generate_report,
)
from neurotrace.storage import TraceDB
from neurotrace.types import (
    LayerSnapshot,
    TokenPrediction,
    TraceMetadata,
    TraceResult,
)


def _make_trace(
    trace_id: str = "test-report-1",
    label: str = "test-report",
    prompt: str = "hello world",
    num_layers: int = 2,
    num_heads: int = 2,
    seq_len: int = 3,
) -> TraceResult:
    """Create a minimal TraceResult for testing."""
    meta = TraceMetadata(
        trace_id=trace_id,
        model_name="test-model",
        model_revision="abc",
        prompt=prompt,
        token_ids=list(range(seq_len)),
        tokens=[f"tok{i}" for i in range(seq_len)],
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_size=4,
        param_count=100,
        device="cpu",
        dtype="float32",
        random_seed=42,
        label=label,
        capture_mode="full",
        layer_stride=1,
        timestamp="2026-03-20T00:00:00",
    )
    snaps = []
    for layer in range(num_layers):
        snaps.append(
            LayerSnapshot(
                layer_index=layer,
                residual_in=np.ones((seq_len, 4), dtype=np.float32),
                residual_out=np.ones((seq_len, 4), dtype=np.float32) * (layer + 1),
                attention_weights=np.random.rand(num_heads, seq_len, seq_len).astype(
                    np.float32
                ),
                attention_output=np.ones((seq_len, 4), dtype=np.float32),
                mlp_in=None,
                mlp_out=None,
                ln_values=None,
                residual_in_norm=2.0 + layer,
                residual_out_norm=2.5 + layer,
                attention_entropy=[0.5 + layer * 0.1] * num_heads,
                mlp_activation_mag=1.0 + layer * 0.5,
                top1_token=42 + layer,
                top1_prob=0.7 - layer * 0.1,
            )
        )
    preds = [
        TokenPrediction(
            position=0,
            top_k_tokens=[42, 43],
            top_k_probs=[0.8, 0.1],
            top_k_strings=["the", "a"],
        )
    ]
    return TraceResult(
        metadata=meta,
        layer_snapshots=snaps,
        token_predictions=preds,
        final_logits=np.zeros((seq_len, 100), dtype=np.float32),
    )


def _make_layer_stats(num_layers: int = 2) -> list[dict]:
    return [
        {
            "layer_index": i,
            "residual_in_norm": 2.0 + i,
            "residual_out_norm": 2.5 + i,
            "attention_entropy": [0.5 + i * 0.1, 0.6 + i * 0.1],
            "mlp_activation_mag": 1.0 + i * 0.5,
            "top1_token": 42 + i,
            "top1_prob": 0.7 - i * 0.1,
        }
        for i in range(num_layers)
    ]


def _make_layer_preds(num_layers: int = 2) -> list[dict]:
    return [
        {
            "layer_index": i,
            "top_k_ids": [42, 43, 44],
            "top_k_probs": [0.7 - i * 0.1, 0.1, 0.05],
            "top_k_strings": ["the", "a", "an"],
            "annotations": ["", "NEW", "--"] if i > 0 else ["", "", ""],
        }
        for i in range(num_layers)
    ]


def _make_token_tracks(num_layers: int = 2) -> list[dict]:
    return [
        {
            "token": "the",
            "token_id": 42,
            "probs": [0.7 - i * 0.05 for i in range(num_layers)],
            "layers": list(range(num_layers)),
        },
        {
            "token": "a",
            "token_id": 43,
            "probs": [0.1 + i * 0.02 for i in range(num_layers)],
            "layers": list(range(num_layers)),
        },
    ]


# --- Unit tests ---


def test_sample_layers_small():
    assert _sample_layers(3) == [0, 1, 2]
    assert _sample_layers(6) == [0, 1, 2, 3, 4, 5]


def test_sample_layers_large():
    result = _sample_layers(22)
    assert 0 in result
    assert 21 in result
    assert len(result) == 6


def test_generate_report_single():
    trace = _make_trace()
    stats = _make_layer_stats()
    preds = _make_layer_preds()
    tracks = _make_token_tracks()

    html = generate_report(trace, stats, preds, tracks)

    assert "<!DOCTYPE html>" in html
    assert "</html>" in html
    assert "NeuroTrace Report" in html
    assert "hello world" in html
    assert "test-model" in html
    assert "Token Prediction Evolution" in html
    assert "Residual Stream Norms" in html
    assert "Attention Entropy" in html
    assert "Attention Heatmaps" in html
    assert "Raw Data" in html
    # Check SVG presence
    assert "<svg" in html
    assert "</svg>" in html


def test_generate_report_no_predictions():
    trace = _make_trace()
    stats = _make_layer_stats()

    html = generate_report(trace, stats, None, None)
    assert "<!DOCTYPE html>" in html
    assert "No prediction data" in html


def test_generate_report_no_attention():
    trace = _make_trace()
    stats = _make_layer_stats()

    html = generate_report(trace, stats, no_attention=True)
    assert "Attention Heatmaps" not in html


def test_generate_report_full_attention():
    trace = _make_trace(num_layers=10)
    stats = _make_layer_stats(num_layers=10)

    html_sampled = generate_report(trace, stats)
    html_full = generate_report(trace, stats, full_attention=True)
    # Full should have more content (more attention layers)
    assert len(html_full) >= len(html_sampled)


def test_generate_comparison_report():
    from neurotrace.analyzer import compute_diff

    trace_a = _make_trace(trace_id="a", label="trace-a", prompt="foo")
    trace_b = _make_trace(trace_id="b", label="trace-b", prompt="bar")
    stats_a = _make_layer_stats()
    stats_b = _make_layer_stats()
    preds_a = _make_layer_preds()
    preds_b = _make_layer_preds()
    tracks_a = _make_token_tracks()
    tracks_b = _make_token_tracks()

    diff_result = compute_diff(trace_a, trace_b)
    token_lookup = {42: "the", 43: "a"}

    html = generate_comparison_report(
        trace_a,
        trace_b,
        stats_a,
        stats_b,
        preds_a,
        preds_b,
        tracks_a,
        tracks_b,
        diff_result,
        token_lookup,
    )

    assert "<!DOCTYPE html>" in html
    assert "</html>" in html
    assert "Comparison Report" in html
    assert "Divergence Summary" in html
    assert "Token Tracking Overlay" in html
    assert "Divergence Heatmap" in html
    assert "trace-a" in html
    assert "trace-b" in html


def test_html_is_well_formed():
    """Basic check: no unclosed tags in the report."""
    trace = _make_trace()
    stats = _make_layer_stats()
    preds = _make_layer_preds()
    tracks = _make_token_tracks()

    html = generate_report(trace, stats, preds, tracks)
    # Check balanced tags for important elements
    assert html.count("<html") == html.count("</html>")
    assert html.count("<head") == html.count("</head>")
    assert html.count("<body") == html.count("</body>")
    assert html.count("<table") == html.count("</table>")
    assert html.count("<svg") == html.count("</svg>")
    assert html.count("<details") == html.count("</details>")


def test_html_special_chars_escaped():
    """Prompts with special chars should be HTML-escaped."""
    trace = _make_trace(prompt='<script>alert("xss")</script>')
    stats = _make_layer_stats()

    html = generate_report(trace, stats)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_report_file_size_reasonable():
    """Single trace report should be under 2MB."""
    trace = _make_trace(num_layers=22, num_heads=4, seq_len=10)
    stats = _make_layer_stats(num_layers=22)
    preds = _make_layer_preds(num_layers=22)
    tracks = _make_token_tracks(num_layers=22)

    html = generate_report(trace, stats, preds, tracks)
    size_kb = len(html.encode("utf-8")) / 1024
    assert size_kb < 2048, f"Report too large: {size_kb:.0f} KB"


# --- CLI integration tests ---


def _write_sample_trace(db_path: str, trace_id: str, label: str) -> None:
    trace = _make_trace(trace_id=trace_id, label=label)
    db = TraceDB(db_path)
    db.write_trace(trace)
    db.close()


def test_report_cli_single_trace(tmp_path):
    db_path = str(tmp_path / "test.db")
    _write_sample_trace(db_path, "rpt-1", "my-trace")

    output = str(tmp_path / "report.html")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "report",
            "--db",
            db_path,
            "--trace-id",
            "my-trace",
            "-o",
            output,
            "--no-attention",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Report written to" in result.output

    with open(output) as f:
        html = f.read()
    assert "<!DOCTYPE html>" in html
    assert "hello world" in html


def test_report_cli_comparison(tmp_path):
    db_path = str(tmp_path / "test.db")
    _write_sample_trace(db_path, "cmp-a", "trace-a")
    _write_sample_trace(db_path, "cmp-b", "trace-b")

    output = str(tmp_path / "compare.html")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "report",
            "--db",
            db_path,
            "--trace-a",
            "trace-a",
            "--trace-b",
            "trace-b",
            "-o",
            output,
            "--no-attention",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Report written to" in result.output

    with open(output) as f:
        html = f.read()
    assert "Comparison Report" in html


def test_report_cli_validation_errors(tmp_path):
    db_path = str(tmp_path / "test.db")
    _write_sample_trace(db_path, "v-1", "v-trace")
    runner = CliRunner()

    # No trace specified
    result = runner.invoke(cli, ["report", "--db", db_path])
    assert result.exit_code != 0

    # Both --trace-id and --trace-a
    result = runner.invoke(
        cli,
        [
            "report",
            "--db",
            db_path,
            "--trace-id",
            "v-trace",
            "--trace-a",
            "v-trace",
        ],
    )
    assert result.exit_code != 0

    # Only --trace-a without --trace-b
    result = runner.invoke(
        cli,
        ["report", "--db", db_path, "--trace-a", "v-trace"],
    )
    assert result.exit_code != 0


def test_report_cli_open_flag(tmp_path):
    """--open flag doesn't crash (we can't verify browser opens)."""
    db_path = str(tmp_path / "test.db")
    _write_sample_trace(db_path, "open-1", "open-trace")

    output = str(tmp_path / "open-report.html")
    runner = CliRunner()
    # Note: --open will try to open a browser, which may silently fail in CI
    # We just verify it doesn't crash the command
    result = runner.invoke(
        cli,
        [
            "report",
            "--db",
            db_path,
            "--trace-id",
            "open-trace",
            "-o",
            output,
            "--no-attention",
            "--open",
        ],
    )
    assert result.exit_code == 0
