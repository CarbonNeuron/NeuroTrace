"""Tests for the upload module."""

import os
from unittest.mock import MagicMock, patch

import pytest
from click import ClickException

from neurotrace.upload import upload_report


def test_missing_env_vars(tmp_path):
    """Missing CF_URL or CF_API_KEY raises ClickException."""
    html = tmp_path / "report.html"
    html.write_text("<html></html>")

    # Mock carbonfiles available but env vars missing
    mock_cf_module = MagicMock()
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.dict("sys.modules", {"carbonfiles": mock_cf_module}),
    ):
        with pytest.raises(ClickException, match="CF_URL and CF_API_KEY"):
            upload_report(str(html))


def test_missing_sdk(tmp_path, monkeypatch):
    """Missing carbonfiles SDK raises helpful ClickException."""
    html = tmp_path / "report.html"
    html.write_text("<html></html>")

    monkeypatch.setenv("CF_URL", "https://example.com")
    monkeypatch.setenv("CF_API_KEY", "test-key")

    # Simulate ImportError by removing carbonfiles from sys.modules
    # and patching builtins.__import__
    original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def mock_import(name, *args, **kwargs):
        if name == "carbonfiles":
            raise ImportError("No module named 'carbonfiles'")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(ClickException, match="CarbonFiles SDK"):
            upload_report(str(html))


def test_bucket_provided_skips_creation(tmp_path, monkeypatch):
    """When --bucket is provided, no bucket creation call is made."""
    html = tmp_path / "report.html"
    html.write_text("<html></html>")

    monkeypatch.setenv("CF_URL", "https://files.example.com")
    monkeypatch.setenv("CF_API_KEY", "test-key")

    mock_cf_class = MagicMock()
    mock_cf_instance = mock_cf_class.return_value
    mock_bucket = MagicMock()
    mock_cf_instance.buckets.__getitem__.return_value = mock_bucket

    mock_cf_module = MagicMock()
    mock_cf_module.CarbonFiles = mock_cf_class

    with patch.dict("sys.modules", {"carbonfiles": mock_cf_module}):
        url = upload_report(str(html), bucket_id="existing-bucket")

    # Should NOT have called buckets.create
    mock_cf_instance.buckets.create.assert_not_called()
    # Should have used the existing bucket
    mock_cf_instance.buckets.__getitem__.assert_called_with("existing-bucket")
    mock_bucket.files.upload.assert_called_once_with(str(html))
    assert url == "https://files.example.com/api/buckets/existing-bucket/files/report.html/content"


def test_successful_upload_creates_bucket(tmp_path, monkeypatch):
    """Without --bucket, creates a new bucket then uploads."""
    html = tmp_path / "report.html"
    html.write_text("<html></html>")

    monkeypatch.setenv("CF_URL", "https://files.example.com")
    monkeypatch.setenv("CF_API_KEY", "test-key")

    mock_cf_class = MagicMock()
    mock_cf_instance = mock_cf_class.return_value
    mock_bucket_obj = MagicMock()
    mock_bucket_obj.id = "new-bucket-123"
    mock_cf_instance.buckets.create.return_value = mock_bucket_obj
    mock_bucket_ref = MagicMock()
    mock_cf_instance.buckets.__getitem__.return_value = mock_bucket_ref

    mock_cf_module = MagicMock()
    mock_cf_module.CarbonFiles = mock_cf_class

    with patch.dict("sys.modules", {"carbonfiles": mock_cf_module}):
        url = upload_report(str(html))

    mock_cf_instance.buckets.create.assert_called_once_with(
        name="neurotrace-report", expires="7d"
    )
    mock_cf_instance.buckets.__getitem__.assert_called_with("new-bucket-123")
    mock_bucket_ref.files.upload.assert_called_once_with(str(html))
    assert url == "https://files.example.com/api/buckets/new-bucket-123/files/report.html/content"


def test_upload_called_after_html_written(tmp_path, monkeypatch):
    """Upload receives the actual file path that was written."""
    html = tmp_path / "my-report.html"
    html.write_text("<!DOCTYPE html><html><body>test</body></html>")

    monkeypatch.setenv("CF_URL", "https://files.example.com")
    monkeypatch.setenv("CF_API_KEY", "key")

    mock_cf_class = MagicMock()
    mock_cf_instance = mock_cf_class.return_value
    mock_bucket_obj = MagicMock()
    mock_bucket_obj.id = "b1"
    mock_cf_instance.buckets.create.return_value = mock_bucket_obj
    mock_bucket_ref = MagicMock()
    mock_cf_instance.buckets.__getitem__.return_value = mock_bucket_ref

    mock_cf_module = MagicMock()
    mock_cf_module.CarbonFiles = mock_cf_class

    with patch.dict("sys.modules", {"carbonfiles": mock_cf_module}):
        url = upload_report(str(html))

    # Verify the exact path was passed
    mock_bucket_ref.files.upload.assert_called_once_with(str(html))
    assert url == "https://files.example.com/api/buckets/b1/files/my-report.html/content"


def test_cli_upload_flag(tmp_path, monkeypatch):
    """CLI --upload flag triggers upload after report generation."""
    from click.testing import CliRunner

    from neurotrace.cli import cli
    from neurotrace.storage import TraceDB
    from neurotrace.types import (
        LayerSnapshot,
        TokenPrediction,
        TraceMetadata,
        TraceResult,
    )
    import numpy as np

    # Write a sample trace
    db_path = str(tmp_path / "test.db")
    meta = TraceMetadata(
        trace_id="upload-test",
        model_name="test-model",
        model_revision="abc",
        prompt="hello",
        token_ids=[1, 2],
        tokens=["hello", " world"],
        num_layers=1,
        num_heads=1,
        hidden_size=4,
        param_count=100,
        device="cpu",
        dtype="float32",
        random_seed=42,
        label="upload-trace",
        capture_mode="full",
        layer_stride=1,
        timestamp="2026-03-20T00:00:00",
    )
    snap = LayerSnapshot(
        layer_index=0,
        residual_in=np.ones((2, 4), dtype=np.float32),
        residual_out=np.ones((2, 4), dtype=np.float32),
        attention_weights=np.ones((1, 2, 2), dtype=np.float32) * 0.5,
        attention_output=np.ones((2, 4), dtype=np.float32),
        mlp_in=None, mlp_out=None, ln_values=None,
        residual_in_norm=2.0, residual_out_norm=2.0,
        attention_entropy=[0.5], mlp_activation_mag=1.0,
        top1_token=42, top1_prob=0.8,
    )
    pred = TokenPrediction(
        position=0, top_k_tokens=[42], top_k_probs=[0.8],
        top_k_strings=["the"],
    )
    trace_result = TraceResult(
        metadata=meta, layer_snapshots=[snap],
        token_predictions=[pred],
        final_logits=np.zeros((2, 100), dtype=np.float32),
    )
    db = TraceDB(db_path)
    db.write_trace(trace_result)
    db.close()

    monkeypatch.setenv("CF_URL", "https://files.example.com")
    monkeypatch.setenv("CF_API_KEY", "test-key")

    output = str(tmp_path / "report.html")
    runner = CliRunner()

    with patch("neurotrace.upload.upload_report") as mock_upload:
        mock_upload.return_value = "https://files.example.com/api/buckets/b1/files/report.html/content"
        result = runner.invoke(
            cli,
            [
                "report", "--db", db_path,
                "--trace-id", "upload-trace",
                "-o", output, "--no-attention", "--upload",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "Uploaded" in result.output
    mock_upload.assert_called_once_with(output, bucket_id=None)
