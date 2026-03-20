"""Tests for the fingerprint module — no model downloads required."""

import json

import numpy as np
import pytest

from neurotrace.fingerprint import (
    AnalyticalRegression,
    Fingerprint,
    FingerprintRun,
    check_regressions_fast,
    compute_alignment_stats,
    deserialize_f16_tensor,
    serialize_f16_tensor,
)
from neurotrace.storage import TraceDB

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_LAYERS = 22
INTERMEDIATE_SIZE = 5632
HIDDEN_SIZE = 2048


def _make_fingerprint(
    prompt="The capital of France is",
    answer="Paris",
    competitor="the",
    margin=4.48,
    seed=42,
) -> Fingerprint:
    rng = np.random.RandomState(seed)
    return Fingerprint(
        prompt=prompt,
        answer=answer,
        competitor=competitor,
        answer_logit=14.21,
        competitor_logit=14.21 - margin,
        margin=margin,
        key_vectors=rng.randn(NUM_LAYERS, INTERMEDIATE_SIZE).astype(np.float16),
        p_answer=rng.randn(HIDDEN_SIZE).astype(np.float16),
        p_competitor=rng.randn(HIDDEN_SIZE).astype(np.float16),
    )


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_roundtrip_2d(self):
        arr = np.random.randn(22, 5632).astype(np.float16)
        blob = serialize_f16_tensor(arr)
        result = deserialize_f16_tensor(blob)
        np.testing.assert_array_equal(result, arr)

    def test_roundtrip_1d(self):
        arr = np.random.randn(2048).astype(np.float16)
        blob = serialize_f16_tensor(arr)
        result = deserialize_f16_tensor(blob)
        np.testing.assert_array_equal(result, arr)

    def test_dtype_preserved(self):
        arr = np.random.randn(10, 20).astype(np.float32)
        blob = serialize_f16_tensor(arr)
        result = deserialize_f16_tensor(blob)
        assert result.dtype == np.float16

    def test_shape_preserved(self):
        arr = np.random.randn(3, 4, 5).astype(np.float16)
        blob = serialize_f16_tensor(arr)
        result = deserialize_f16_tensor(blob)
        assert result.shape == (3, 4, 5)


# ---------------------------------------------------------------------------
# Data structure tests
# ---------------------------------------------------------------------------


class TestFingerprintDataStructures:
    def test_fingerprint_fields(self):
        fp = _make_fingerprint()
        assert fp.prompt == "The capital of France is"
        assert fp.answer == "Paris"
        assert fp.competitor == "the"
        assert fp.margin == pytest.approx(4.48)
        assert fp.key_vectors.shape == (NUM_LAYERS, INTERMEDIATE_SIZE)
        assert fp.p_answer.shape == (HIDDEN_SIZE,)
        assert fp.p_competitor.shape == (HIDDEN_SIZE,)
        assert fp.key_vectors.dtype == np.float16
        assert fp.p_answer.dtype == np.float16

    def test_fingerprint_run(self):
        fps = [_make_fingerprint(), _make_fingerprint(answer="Berlin")]
        run = FingerprintRun(
            run_id="fp1",
            dataset="capitals",
            model_name="test-model",
            prompt_count=2,
            fingerprints=fps,
        )
        assert run.prompt_count == 2
        assert len(run.fingerprints) == 2

    def test_analytical_regression(self):
        ar = AnalyticalRegression(
            prompt="test",
            answer="ans",
            current_margin=4.0,
            margin_change=-5.0,
            new_margin=-1.0,
            alignment=0.85,
            regression=True,
        )
        assert ar.regression is True
        assert ar.new_margin == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Analytical regression check tests
# ---------------------------------------------------------------------------


class TestCheckRegressionsfast:
    def test_orthogonal_no_impact(self):
        """Orthogonal key vectors should produce no regression."""
        # Create k_star and fingerprint with orthogonal key vectors
        k_star = np.zeros(INTERMEDIATE_SIZE, dtype=np.float32)
        k_star[0] = 1.0  # unit vector along dim 0

        fp = _make_fingerprint()
        # Make fp key vector orthogonal to k_star at the target layer
        fp.key_vectors[5] = np.zeros(INTERMEDIATE_SIZE, dtype=np.float16)
        fp.key_vectors[5, 1] = 1.0  # unit vector along dim 1

        delta = np.ones(HIDDEN_SIZE, dtype=np.float32)

        affected = check_regressions_fast(
            delta, k_star, layer=5, fingerprints=[fp],
        )
        assert len(affected) == 0

    def test_aligned_creates_impact(self):
        """Aligned key vectors should produce detectable impact."""
        k_star = np.ones(INTERMEDIATE_SIZE, dtype=np.float32)
        delta = np.ones(HIDDEN_SIZE, dtype=np.float32) * 10.0

        fp = _make_fingerprint(margin=0.5)
        # Make fp key vector aligned with k_star
        fp.key_vectors[5] = np.ones(INTERMEDIATE_SIZE, dtype=np.float16)
        # Make p_answer - p_competitor aligned with delta
        fp.p_answer = np.ones(HIDDEN_SIZE, dtype=np.float16)
        fp.p_competitor = np.zeros(HIDDEN_SIZE, dtype=np.float16)

        affected = check_regressions_fast(
            delta, k_star, layer=5, fingerprints=[fp], threshold=0.01,
        )
        assert len(affected) == 1
        assert affected[0].alignment == pytest.approx(1.0, abs=0.01)

    def test_regression_detection(self):
        """A repair that would flip margin from positive to negative."""
        k_star = np.ones(INTERMEDIATE_SIZE, dtype=np.float32)
        # Large negative delta should cause regression
        delta = -np.ones(HIDDEN_SIZE, dtype=np.float32) * 100.0

        fp = _make_fingerprint(margin=0.1)  # barely positive
        fp.key_vectors[5] = np.ones(INTERMEDIATE_SIZE, dtype=np.float16)
        fp.p_answer = np.ones(HIDDEN_SIZE, dtype=np.float16)
        fp.p_competitor = np.zeros(HIDDEN_SIZE, dtype=np.float16)

        affected = check_regressions_fast(
            delta, k_star, layer=5, fingerprints=[fp], threshold=0.01,
        )
        assert len(affected) == 1
        assert affected[0].regression is True
        assert affected[0].new_margin < 0

    def test_threshold_filtering(self):
        """Changes below threshold should be filtered out."""
        k_star = np.ones(INTERMEDIATE_SIZE, dtype=np.float32) * 0.001
        delta = np.ones(HIDDEN_SIZE, dtype=np.float32) * 0.001

        fp = _make_fingerprint()
        fp.key_vectors[5] = np.ones(INTERMEDIATE_SIZE, dtype=np.float16) * 0.001
        fp.p_answer = np.ones(HIDDEN_SIZE, dtype=np.float16) * 0.001
        fp.p_competitor = np.zeros(HIDDEN_SIZE, dtype=np.float16)

        affected = check_regressions_fast(
            delta, k_star, layer=5, fingerprints=[fp], threshold=100.0,
        )
        assert len(affected) == 0

    def test_zero_k_star(self):
        """Zero k_star should produce no results."""
        k_star = np.zeros(INTERMEDIATE_SIZE, dtype=np.float32)
        delta = np.ones(HIDDEN_SIZE, dtype=np.float32)
        fp = _make_fingerprint()
        affected = check_regressions_fast(
            delta, k_star, layer=5, fingerprints=[fp],
        )
        assert len(affected) == 0

    def test_multiple_fingerprints(self):
        """Test with multiple fingerprints."""
        k_star = np.ones(INTERMEDIATE_SIZE, dtype=np.float32)
        delta = np.ones(HIDDEN_SIZE, dtype=np.float32) * 5.0

        fps = []
        for i in range(5):
            fp = _make_fingerprint(
                prompt=f"prompt {i}",
                answer=f"answer{i}",
                margin=float(i),
                seed=i,
            )
            fp.key_vectors[3] = np.ones(INTERMEDIATE_SIZE, dtype=np.float16)
            fp.p_answer = np.ones(HIDDEN_SIZE, dtype=np.float16)
            fp.p_competitor = np.zeros(HIDDEN_SIZE, dtype=np.float16)
            fps.append(fp)

        affected = check_regressions_fast(
            delta, k_star, layer=3, fingerprints=fps, threshold=0.01,
        )
        # All should be affected since they're all aligned
        assert len(affected) == 5


# ---------------------------------------------------------------------------
# Alignment stats tests
# ---------------------------------------------------------------------------


class TestAlignmentStats:
    def test_single_fingerprint(self):
        stats = compute_alignment_stats([_make_fingerprint()])
        assert stats["avg_alignment"] == 0.0
        assert stats["max_pair"] is None

    def test_two_fingerprints(self):
        fp1 = _make_fingerprint(seed=1)
        fp2 = _make_fingerprint(answer="Berlin", seed=2)
        stats = compute_alignment_stats([fp1, fp2])
        assert 0 <= stats["avg_alignment"] <= 1
        assert stats["max_pair"] is not None

    def test_identical_fingerprints(self):
        fp1 = _make_fingerprint(seed=42)
        fp2 = _make_fingerprint(answer="Berlin", seed=42)
        stats = compute_alignment_stats([fp1, fp2])
        # Same random seed → same vectors → alignment should be ~1.0
        assert stats["max_alignment"] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Storage tests
# ---------------------------------------------------------------------------


class TestFingerprintStorage:
    def test_write_and_list(self, tmp_path):
        db = TraceDB(str(tmp_path / "test.db"))
        db.write_fingerprint_run(
            run_id="fp1", dataset="capitals",
            model_name="test", prompt_count=5,
        )
        runs = db.list_fingerprint_runs()
        assert len(runs) == 1
        assert runs[0]["id"] == "fp1"
        assert runs[0]["dataset"] == "capitals"
        db.close()

    def test_write_and_load_fingerprints(self, tmp_path):
        db = TraceDB(str(tmp_path / "test.db"))
        db.write_fingerprint_run(
            run_id="fp1", dataset="capitals",
            model_name="test", prompt_count=1,
        )

        fp = _make_fingerprint()
        db.write_fingerprint(
            run_id="fp1",
            prompt=fp.prompt,
            answer=fp.answer,
            competitor=fp.competitor,
            answer_logit=fp.answer_logit,
            competitor_logit=fp.competitor_logit,
            margin=fp.margin,
            key_vectors_blob=serialize_f16_tensor(fp.key_vectors),
            p_answer_blob=serialize_f16_tensor(fp.p_answer),
            p_competitor_blob=serialize_f16_tensor(fp.p_competitor),
        )

        loaded = db.load_fingerprints("fp1")
        assert len(loaded) == 1
        assert loaded[0].prompt == "The capital of France is"
        assert loaded[0].answer == "Paris"
        assert loaded[0].margin == pytest.approx(4.48, abs=0.01)
        np.testing.assert_array_almost_equal(
            loaded[0].key_vectors, fp.key_vectors, decimal=2,
        )
        np.testing.assert_array_almost_equal(
            loaded[0].p_answer, fp.p_answer, decimal=2,
        )
        db.close()

    def test_get_latest_run_id(self, tmp_path):
        db = TraceDB(str(tmp_path / "test.db"))
        db.write_fingerprint_run(
            run_id="fp1", dataset="caps1",
            model_name="test", prompt_count=5,
        )
        db.write_fingerprint_run(
            run_id="fp2", dataset="caps2",
            model_name="test", prompt_count=10,
        )
        latest = db.get_latest_fingerprint_run_id()
        assert latest == "fp2"
        db.close()

    def test_get_latest_run_id_empty(self, tmp_path):
        db = TraceDB(str(tmp_path / "test.db"))
        with pytest.raises(ValueError, match="No fingerprint runs"):
            db.get_latest_fingerprint_run_id()
        db.close()

    def test_multiple_fingerprints_in_run(self, tmp_path):
        db = TraceDB(str(tmp_path / "test.db"))
        db.write_fingerprint_run(
            run_id="fp1", dataset="capitals",
            model_name="test", prompt_count=3,
        )

        for i, (prompt, answer) in enumerate([
            ("France", "Paris"),
            ("Germany", "Berlin"),
            ("Japan", "Tokyo"),
        ]):
            fp = _make_fingerprint(prompt=prompt, answer=answer, seed=i)
            db.write_fingerprint(
                run_id="fp1",
                prompt=fp.prompt,
                answer=fp.answer,
                competitor=fp.competitor,
                answer_logit=fp.answer_logit,
                competitor_logit=fp.competitor_logit,
                margin=fp.margin,
                key_vectors_blob=serialize_f16_tensor(fp.key_vectors),
                p_answer_blob=serialize_f16_tensor(fp.p_answer),
                p_competitor_blob=serialize_f16_tensor(fp.p_competitor),
            )

        loaded = db.load_fingerprints("fp1")
        assert len(loaded) == 3
        assert loaded[0].answer == "Paris"
        assert loaded[1].answer == "Berlin"
        assert loaded[2].answer == "Tokyo"
        db.close()


# ---------------------------------------------------------------------------
# Build from remote tests
# ---------------------------------------------------------------------------


class TestBuildFromRemote:
    def test_build_fingerprint_from_remote(self):
        import base64

        from neurotrace.fingerprint import build_fingerprint_from_remote

        kv = np.random.randn(22, 5632).astype(np.float16)
        pa = np.random.randn(2048).astype(np.float16)
        pc = np.random.randn(2048).astype(np.float16)

        data = {
            "prompt": "The capital of France is",
            "answer": "Paris",
            "competitor": "the",
            "answer_logit": 14.21,
            "competitor_logit": 9.73,
            "margin": 4.48,
            "key_vectors_b64": base64.b64encode(kv.tobytes()).decode(),
            "key_vectors_shape": list(kv.shape),
            "p_answer_b64": base64.b64encode(pa.tobytes()).decode(),
            "p_competitor_b64": base64.b64encode(pc.tobytes()).decode(),
        }

        fp = build_fingerprint_from_remote(data)
        assert fp.prompt == "The capital of France is"
        assert fp.answer == "Paris"
        assert fp.margin == pytest.approx(4.48)
        assert fp.key_vectors.shape == (22, 5632)
        assert fp.p_answer.shape == (2048,)
        np.testing.assert_array_almost_equal(fp.key_vectors, kv, decimal=2)


# ---------------------------------------------------------------------------
# CLI integration tests (require model download)
# ---------------------------------------------------------------------------


@pytest.mark.model_download
class TestFingerprintCLI:
    def test_fingerprint_local(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "fingerprint",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--dataset-builtin", "capitals",
        ])
        assert result.exit_code == 0, result.output
        assert "Fingerprint" in result.output
        assert "fingerprints" in result.output.lower()

        # Check DB was populated
        db = TraceDB(db_path)
        runs = db.list_fingerprint_runs()
        assert len(runs) == 1
        assert runs[0]["prompt_count"] == 50

        fps = db.load_fingerprints(runs[0]["id"])
        assert len(fps) == 50
        assert fps[0].key_vectors.shape[0] == 22  # TinyLlama layers
        db.close()

    def test_fingerprint_json(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "fingerprint",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--dataset-builtin", "capitals",
            "--json",
        ])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output.strip())
        assert "fingerprints" in data
        assert data["prompt_count"] == 50

    def test_fingerprint_list(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        runner = CliRunner()

        # First create fingerprints
        runner.invoke(cli, [
            "fingerprint",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--dataset-builtin", "capitals",
        ])

        # Then list
        result = runner.invoke(cli, [
            "fingerprint",
            "--db", db_path,
            "--list",
        ])
        assert result.exit_code == 0, result.output
        assert "capitals" in result.output
