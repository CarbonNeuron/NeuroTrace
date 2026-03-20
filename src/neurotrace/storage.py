"""DuckDB trace storage — read, write, and query traces."""

import struct

import duckdb
import numpy as np

from neurotrace.types import (
    LayerSnapshot,
    TokenPrediction,
    TraceMetadata,
    TraceResult,
)


def _serialize_tensor(arr: np.ndarray) -> bytes:
    """Serialize numpy array to bytes with shape header.

    Format: [ndim as int32] [dim0 as int32] [dim1 as int32] ... [raw float32 data]
    """
    header = struct.pack("<i", arr.ndim)
    for dim in arr.shape:
        header += struct.pack("<i", dim)
    return header + arr.astype(np.float32).tobytes()


def _deserialize_tensor(blob: bytes) -> np.ndarray:
    """Deserialize bytes back to numpy array."""
    offset = 0
    ndim = struct.unpack_from("<i", blob, offset)[0]
    offset += 4
    shape = []
    for _ in range(ndim):
        shape.append(struct.unpack_from("<i", blob, offset)[0])
        offset += 4
    data = np.frombuffer(blob, dtype=np.float32, offset=offset)
    return data.reshape(shape)


class TraceDB:
    """DuckDB-backed trace storage."""

    def __init__(self, db_path: str):
        self._conn = duckdb.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                trace_id VARCHAR PRIMARY KEY,
                model_name VARCHAR NOT NULL,
                model_revision VARCHAR,
                prompt VARCHAR NOT NULL,
                token_ids INTEGER[],
                tokens VARCHAR[],
                num_layers INTEGER NOT NULL,
                num_heads INTEGER NOT NULL,
                hidden_size INTEGER NOT NULL,
                param_count BIGINT,
                device VARCHAR,
                dtype VARCHAR,
                random_seed INTEGER,
                label VARCHAR,
                capture_mode VARCHAR,
                layer_stride INTEGER,
                top_prediction VARCHAR,
                top_prediction_prob FLOAT,
                timestamp VARCHAR NOT NULL,
                interventions VARCHAR
            )
        """)
        # Migration: add interventions column if missing (for pre-existing DBs)
        try:
            self._conn.execute("ALTER TABLE traces ADD COLUMN interventions VARCHAR")
        except duckdb.CatalogException:
            pass  # column already exists

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS layer_snapshots (
                trace_id VARCHAR NOT NULL,
                layer_index INTEGER NOT NULL,
                residual_in_norm FLOAT,
                residual_out_norm FLOAT,
                attention_entropy FLOAT[],
                mlp_activation_mag FLOAT,
                top1_token INTEGER,
                top1_prob FLOAT,
                residual_in_blob BLOB,
                residual_out_blob BLOB,
                attention_output_blob BLOB,
                mlp_in_blob BLOB,
                mlp_out_blob BLOB,
                ln_values_blob BLOB,
                PRIMARY KEY (trace_id, layer_index)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS attention_maps (
                trace_id VARCHAR NOT NULL,
                layer_index INTEGER NOT NULL,
                head_index INTEGER NOT NULL,
                entropy FLOAT,
                weights_blob BLOB NOT NULL,
                PRIMARY KEY (trace_id, layer_index, head_index)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS token_predictions (
                trace_id VARCHAR NOT NULL,
                position INTEGER NOT NULL,
                top_k_tokens INTEGER[],
                top_k_probs FLOAT[],
                top_k_strings VARCHAR[],
                PRIMARY KEY (trace_id, position)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS finetune_runs (
                id VARCHAR PRIMARY KEY,
                model VARCHAR NOT NULL,
                adapter_path VARCHAR NOT NULL,
                target_layers VARCHAR NOT NULL,
                lora_rank INTEGER NOT NULL,
                lora_alpha INTEGER NOT NULL,
                dataset_name VARCHAR,
                dataset_size INTEGER,
                epochs INTEGER NOT NULL,
                learning_rate REAL NOT NULL,
                seed INTEGER,
                train_loss_start REAL,
                train_loss_end REAL,
                scan_before VARCHAR,
                scan_after VARCHAR,
                created_at VARCHAR DEFAULT (CURRENT_TIMESTAMP)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                dataset_name TEXT NOT NULL,
                model TEXT NOT NULL,
                baseline_correct INTEGER,
                baseline_sabotaged INTEGER,
                baseline_weak INTEGER,
                baseline_wrong INTEGER,
                target_layers VARCHAR,
                finetune_run_id TEXT,
                adapter_path TEXT,
                result_correct INTEGER,
                result_sabotaged INTEGER,
                result_weak INTEGER,
                result_wrong INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                duration_seconds REAL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS neuron_profiles (
                id VARCHAR PRIMARY KEY,
                trace_id VARCHAR,
                contrast_trace_id VARCHAR,
                layer INTEGER NOT NULL,
                component VARCHAR NOT NULL,
                prompt VARCHAR NOT NULL,
                contrast_prompt VARCHAR,
                model VARCHAR NOT NULL,
                neuron_indices VARCHAR NOT NULL,
                target_activations VARCHAR NOT NULL,
                contrast_activations VARCHAR,
                diff_activations VARCHAR,
                label VARCHAR,
                created_at VARCHAR
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS probes (
                id TEXT PRIMARY KEY,
                dataset_name TEXT NOT NULL,
                model TEXT NOT NULL,
                layer INTEGER NOT NULL,
                extraction_point TEXT NOT NULL,
                num_clean INTEGER,
                num_sabotaged INTEGER,
                cohens_d REAL,
                auc_roc REAL,
                probe_accuracy REAL,
                direction_alignment REAL,
                cross_dataset TEXT,
                cross_auc_roc REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS circuits (
                id TEXT PRIMARY KEY,
                probe_id TEXT,
                model TEXT NOT NULL,
                layer INTEGER NOT NULL,
                top_boosted_tokens TEXT,
                top_suppressed_tokens TEXT,
                reverse_tokens TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS heatmap_runs (
                run_id TEXT PRIMARY KEY,
                dataset_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                adapter_path TEXT,
                num_layers INTEGER NOT NULL,
                num_prompts INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                cells JSON NOT NULL,
                summaries JSON NOT NULL
            )
        """)

    def write_trace(
        self, result: TraceResult, interventions: str | None = None
    ) -> None:
        """Write a complete trace to the database."""
        meta = result.metadata

        # Determine top prediction from token_predictions at last position
        top_pred = ""
        top_pred_prob = 0.0
        if result.token_predictions:
            last = result.token_predictions[-1]
            if last.top_k_strings:
                top_pred = last.top_k_strings[0]
                top_pred_prob = last.top_k_probs[0]

        self._conn.execute(
            "INSERT INTO traces VALUES"
            " (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                meta.trace_id,
                meta.model_name,
                meta.model_revision,
                meta.prompt,
                meta.token_ids,
                meta.tokens,
                meta.num_layers,
                meta.num_heads,
                meta.hidden_size,
                meta.param_count,
                meta.device,
                meta.dtype,
                meta.random_seed,
                meta.label,
                meta.capture_mode,
                meta.layer_stride,
                top_pred,
                top_pred_prob,
                meta.timestamp,
                interventions,
            ],
        )

        for snap in result.layer_snapshots:
            self._conn.execute(
                "INSERT INTO layer_snapshots VALUES"
                " (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    meta.trace_id,
                    snap.layer_index,
                    snap.residual_in_norm,
                    snap.residual_out_norm,
                    snap.attention_entropy,
                    snap.mlp_activation_mag,
                    snap.top1_token,
                    snap.top1_prob,
                    _serialize_tensor(snap.residual_in)
                    if snap.residual_in is not None
                    else None,
                    _serialize_tensor(snap.residual_out)
                    if snap.residual_out is not None
                    else None,
                    _serialize_tensor(snap.attention_output)
                    if snap.attention_output is not None
                    else None,
                    _serialize_tensor(snap.mlp_in) if snap.mlp_in is not None else None,
                    _serialize_tensor(snap.mlp_out)
                    if snap.mlp_out is not None
                    else None,
                    _serialize_tensor(snap.ln_values)
                    if snap.ln_values is not None
                    else None,
                ],
            )

            # Write per-head attention maps
            if snap.attention_weights is not None:
                num_heads = snap.attention_weights.shape[0]
                for h in range(num_heads):
                    head_weights = snap.attention_weights[h]  # [seq, seq]
                    head_entropy = (
                        snap.attention_entropy[h]
                        if h < len(snap.attention_entropy)
                        else 0.0
                    )
                    self._conn.execute(
                        """INSERT INTO attention_maps VALUES (?, ?, ?, ?, ?)""",
                        [
                            meta.trace_id,
                            snap.layer_index,
                            h,
                            head_entropy,
                            _serialize_tensor(head_weights),
                        ],
                    )

        for pred in result.token_predictions:
            self._conn.execute(
                """INSERT INTO token_predictions VALUES (?, ?, ?, ?, ?)""",
                [
                    meta.trace_id,
                    pred.position,
                    pred.top_k_tokens,
                    pred.top_k_probs,
                    pred.top_k_strings,
                ],
            )

    def resolve_trace_id(self, identifier: str) -> str:
        """Resolve a label, prefix, or full trace_id to a full trace_id."""
        if identifier == "latest":
            return self.get_latest_trace_id()

        # Try exact trace_id match
        row = self._conn.execute(
            "SELECT trace_id FROM traces WHERE trace_id = ?",
            [identifier],
        ).fetchone()
        if row:
            return row[0]

        # Try label match
        rows = self._conn.execute(
            "SELECT trace_id FROM traces WHERE label = ?",
            [identifier],
        ).fetchall()
        if len(rows) == 1:
            return rows[0][0]
        if len(rows) > 1:
            raise ValueError(
                f"Ambiguous label {identifier!r}: matches {len(rows)} traces"
            )

        # Try prefix match on trace_id
        rows = self._conn.execute(
            "SELECT trace_id FROM traces WHERE trace_id LIKE ? || '%'",
            [identifier],
        ).fetchall()
        if len(rows) == 1:
            return rows[0][0]
        if len(rows) > 1:
            raise ValueError(
                f"Ambiguous prefix {identifier!r}: matches {len(rows)} traces"
            )

        raise ValueError(f"Trace not found: {identifier}")

    def read_trace(self, trace_id: str) -> TraceResult:
        """Read a complete trace from the database."""
        trace_id = self.resolve_trace_id(trace_id)
        row = self._conn.execute(
            "SELECT * FROM traces WHERE trace_id = ?", [trace_id]
        ).fetchone()
        if row is None:
            raise ValueError(f"Trace not found: {trace_id}")

        meta = TraceMetadata(
            trace_id=row[0],
            model_name=row[1],
            model_revision=row[2],
            prompt=row[3],
            token_ids=row[4],
            tokens=row[5],
            num_layers=row[6],
            num_heads=row[7],
            hidden_size=row[8],
            param_count=row[9],
            device=row[10],
            dtype=row[11],
            random_seed=row[12],
            label=row[13],
            capture_mode=row[14],
            layer_stride=row[15],
            timestamp=row[18],
        )

        snap_rows = self._conn.execute(
            "SELECT * FROM layer_snapshots WHERE trace_id = ? ORDER BY layer_index",
            [trace_id],
        ).fetchall()

        snapshots = []
        for sr in snap_rows:
            # Reconstruct attention weights from per-head maps
            head_rows = self._conn.execute(
                "SELECT head_index, weights_blob FROM attention_maps"
                " WHERE trace_id = ? AND layer_index = ?"
                " ORDER BY head_index",
                [trace_id, sr[1]],
            ).fetchall()
            attn_weights = None
            if head_rows:
                heads = [_deserialize_tensor(hr[1]) for hr in head_rows]
                attn_weights = np.stack(heads, axis=0)

            snapshots.append(
                LayerSnapshot(
                    layer_index=sr[1],
                    residual_in=_deserialize_tensor(sr[8])
                    if sr[8] is not None
                    else None,
                    residual_out=_deserialize_tensor(sr[9])
                    if sr[9] is not None
                    else None,
                    attention_weights=attn_weights,
                    attention_output=_deserialize_tensor(sr[10])
                    if sr[10] is not None
                    else None,
                    mlp_in=_deserialize_tensor(sr[11]) if sr[11] is not None else None,
                    mlp_out=_deserialize_tensor(sr[12]) if sr[12] is not None else None,
                    ln_values=_deserialize_tensor(sr[13])
                    if sr[13] is not None
                    else None,
                    residual_in_norm=sr[2],
                    residual_out_norm=sr[3],
                    attention_entropy=sr[4] if sr[4] is not None else [],
                    mlp_activation_mag=sr[5],
                    top1_token=sr[6],
                    top1_prob=sr[7],
                )
            )

        pred_rows = self._conn.execute(
            "SELECT * FROM token_predictions WHERE trace_id = ? ORDER BY position",
            [trace_id],
        ).fetchall()

        predictions = [
            TokenPrediction(
                position=pr[1],
                top_k_tokens=pr[2],
                top_k_probs=pr[3],
                top_k_strings=pr[4],
            )
            for pr in pred_rows
        ]

        # final_logits not stored in DB — return empty array
        vocab_estimate = 32000
        seq_len = len(meta.token_ids)
        final_logits = np.zeros((seq_len, vocab_estimate), dtype=np.float32)

        return TraceResult(
            metadata=meta,
            layer_snapshots=snapshots,
            token_predictions=predictions,
            final_logits=final_logits,
        )

    def list_traces(self, model_filter: str | None = None) -> list[dict]:
        """List all traces, optionally filtered by model name."""
        query = (
            "SELECT trace_id, label, model_name, prompt,"
            " top_prediction, top_prediction_prob, timestamp"
            " FROM traces"
        )
        params = []
        if model_filter:
            query += " WHERE model_name LIKE ?"
            params.append(f"%{model_filter}%")
        query += " ORDER BY timestamp DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [
            {
                "trace_id": r[0],
                "label": r[1],
                "model_name": r[2],
                "prompt": r[3],
                "top_prediction": r[4],
                "top_prediction_prob": r[5],
                "timestamp": r[6],
            }
            for r in rows
        ]

    def get_latest_trace_id(self) -> str:
        """Return the trace_id of the most recent trace."""
        row = self._conn.execute(
            "SELECT trace_id FROM traces ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        if row is None:
            raise ValueError("No traces in database")
        return row[0]

    def get_attention_map(
        self, trace_id: str, layer_index: int, head_index: int
    ) -> np.ndarray:
        """Get a single head's attention weight matrix."""
        row = self._conn.execute(
            "SELECT weights_blob FROM attention_maps"
            " WHERE trace_id = ? AND layer_index = ?"
            " AND head_index = ?",
            [trace_id, layer_index, head_index],
        ).fetchone()
        if row is None:
            raise ValueError(
                f"Attention map not found: trace={trace_id},"
                f" layer={layer_index}, head={head_index}"
            )
        return _deserialize_tensor(row[0])

    def get_layer_stats(self, trace_id: str) -> list[dict]:
        """Get pre-computed stats for all layers (no blob deserialization)."""
        rows = self._conn.execute(
            """SELECT layer_index, residual_in_norm, residual_out_norm,
                      attention_entropy, mlp_activation_mag, top1_token, top1_prob
               FROM layer_snapshots WHERE trace_id = ? ORDER BY layer_index""",
            [trace_id],
        ).fetchall()
        return [
            {
                "layer_index": r[0],
                "residual_in_norm": r[1],
                "residual_out_norm": r[2],
                "attention_entropy": r[3],
                "mlp_activation_mag": r[4],
                "top1_token": r[5],
                "top1_prob": r[6],
            }
            for r in rows
        ]

    def find_existing_trace(
        self,
        model_name: str,
        prompt: str,
        seed: int,
        capture_mode: str,
    ) -> str | None:
        """Find an existing trace matching these parameters."""
        row = self._conn.execute(
            "SELECT trace_id FROM traces"
            " WHERE model_name = ? AND prompt = ?"
            " AND random_seed = ? AND capture_mode = ?",
            [model_name, prompt, seed, capture_mode],
        ).fetchone()
        return row[0] if row else None

    def get_interventions(self, trace_id: str) -> str | None:
        """Get the interventions JSON for a trace, or None."""
        row = self._conn.execute(
            "SELECT interventions FROM traces WHERE trace_id = ?",
            [trace_id],
        ).fetchone()
        if row is None:
            return None
        return row[0]

    def save_neuron_profile(self, profile) -> None:
        """Save a NeuronProfile to the database."""
        import json

        self._conn.execute(
            "INSERT INTO neuron_profiles VALUES"
            " (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                profile.profile_id,
                profile.trace_id,
                profile.contrast_trace_id,
                profile.layer,
                profile.component,
                profile.prompt,
                profile.contrast_prompt,
                profile.model_name,
                json.dumps(profile.neuron_indices),
                json.dumps(profile.target_activations),
                json.dumps(profile.contrast_activations)
                if profile.contrast_activations
                else None,
                json.dumps(profile.diff_activations)
                if profile.diff_activations
                else None,
                profile.label,
                profile.created_at,
            ],
        )

    def load_neuron_profile(self, label: str):
        """Load a NeuronProfile by label. Returns None if not found."""
        import json

        from neurotrace.neurons import NeuronProfile

        row = self._conn.execute(
            "SELECT * FROM neuron_profiles"
            " WHERE label = ? ORDER BY created_at DESC LIMIT 1",
            [label],
        ).fetchone()
        if row is None:
            # Try by ID
            row = self._conn.execute(
                "SELECT * FROM neuron_profiles WHERE id = ?",
                [label],
            ).fetchone()
        if row is None:
            return None

        return NeuronProfile(
            profile_id=row[0],
            trace_id=row[1],
            contrast_trace_id=row[2],
            layer=row[3],
            component=row[4],
            prompt=row[5],
            contrast_prompt=row[6],
            model_name=row[7],
            neuron_indices=json.loads(row[8]),
            target_activations=json.loads(row[9]),
            contrast_activations=json.loads(row[10]) if row[10] else None,
            diff_activations=json.loads(row[11]) if row[11] else None,
            label=row[12],
            created_at=row[13],
        )

    def save_finetune_run(self, result) -> None:
        """Save a FinetuneResult to the database."""
        import json as json_mod

        self._conn.execute(
            "INSERT INTO finetune_runs VALUES"
            " (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                result.run_id,
                result.model_name,
                result.adapter_path,
                json_mod.dumps(result.target_layers),
                result.lora_rank,
                result.lora_alpha,
                result.dataset_name,
                result.dataset_size,
                result.epochs,
                result.learning_rate,
                result.seed,
                result.train_loss_start,
                result.train_loss_end,
                json_mod.dumps(result.scan_before) if result.scan_before else None,
                json_mod.dumps(result.scan_after) if result.scan_after else None,
                result.created_at,
            ],
        )

    def load_finetune_run(self, run_id: str):
        """Load a FinetuneResult by ID. Returns None if not found."""
        import json as json_mod

        from neurotrace.finetune import FinetuneResult

        row = self._conn.execute(
            "SELECT * FROM finetune_runs WHERE id = ?",
            [run_id],
        ).fetchone()
        if row is None:
            return None

        return FinetuneResult(
            run_id=row[0],
            model_name=row[1],
            adapter_path=row[2],
            target_layers=json_mod.loads(row[3]),
            lora_rank=row[4],
            lora_alpha=row[5],
            dataset_name=row[6],
            dataset_size=row[7],
            epochs=row[8],
            learning_rate=row[9],
            seed=row[10],
            train_loss_start=row[11],
            train_loss_end=row[12],
            scan_before=json_mod.loads(row[13]) if row[13] else None,
            scan_after=json_mod.loads(row[14]) if row[14] else None,
            created_at=row[15],
        )

    def save_experiment(self, experiment: dict) -> None:
        """Save an experiment record to the database."""
        self._conn.execute(
            "INSERT INTO experiments VALUES"
            " (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                experiment["id"],
                experiment["dataset_name"],
                experiment["model"],
                experiment.get("baseline_correct"),
                experiment.get("baseline_sabotaged"),
                experiment.get("baseline_weak"),
                experiment.get("baseline_wrong"),
                experiment.get("target_layers"),
                experiment.get("finetune_run_id"),
                experiment.get("adapter_path"),
                experiment.get("result_correct"),
                experiment.get("result_sabotaged"),
                experiment.get("result_weak"),
                experiment.get("result_wrong"),
                experiment.get("created_at"),
                experiment.get("duration_seconds"),
            ],
        )

    def load_experiment(self, experiment_id: str) -> dict | None:
        """Load an experiment record by ID."""
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE id = ?",
            [experiment_id],
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "dataset_name": row[1],
            "model": row[2],
            "baseline_correct": row[3],
            "baseline_sabotaged": row[4],
            "baseline_weak": row[5],
            "baseline_wrong": row[6],
            "target_layers": row[7],
            "finetune_run_id": row[8],
            "adapter_path": row[9],
            "result_correct": row[10],
            "result_sabotaged": row[11],
            "result_weak": row[12],
            "result_wrong": row[13],
            "created_at": row[14],
            "duration_seconds": row[15],
        }

    def save_probe(self, probe_id: str, result) -> None:
        """Save a probe analysis result to the database."""
        self._conn.execute(
            "INSERT INTO probes VALUES"
            " (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            [
                probe_id,
                result.dataset_name,
                result.model_name,
                result.layer,
                result.extraction_point,
                result.num_clean,
                result.num_sabotaged,
                result.cohens_d,
                result.auc_roc,
                result.probe_accuracy,
                result.direction_alignment,
                result.cross_dataset,
                result.cross_auc_roc,
            ],
        )

    def save_circuit(self, circuit_id: str, result, probe_id: str | None = None) -> None:
        """Save a circuit analysis result to the database."""
        import json

        top_boosted = json.dumps(result.forward.top_boosted)
        top_suppressed = json.dumps(result.forward.top_suppressed)
        reverse_tokens = json.dumps([
            {"token": r.token, "cosine_sim": r.cosine_sim_with_probe}
            for r in result.reverse
        ]) if result.reverse else None

        self._conn.execute(
            "INSERT INTO circuits VALUES"
            " (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            [
                circuit_id,
                probe_id,
                result.model_name,
                result.layer,
                top_boosted,
                top_suppressed,
                reverse_tokens,
            ],
        )

    def write_heatmap_run(
        self,
        run_id: str,
        dataset_name: str,
        model_name: str,
        num_layers: int,
        num_prompts: int,
        cells_json: str,
        summaries_json: str,
        adapter_path: str | None = None,
    ) -> None:
        """Write a heatmap run to the database."""
        self._conn.execute(
            "INSERT INTO heatmap_runs VALUES"
            " (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)",
            [
                run_id,
                dataset_name,
                model_name,
                adapter_path,
                num_layers,
                num_prompts,
                cells_json,
                summaries_json,
            ],
        )

    def read_heatmap_run(self, run_id: str) -> dict:
        """Read a heatmap run from the database."""
        row = self._conn.execute(
            "SELECT * FROM heatmap_runs WHERE run_id = ?", [run_id]
        ).fetchone()
        if row is None:
            raise ValueError(f"Heatmap run not found: {run_id}")
        cols = [
            "run_id", "dataset_name", "model_name", "adapter_path",
            "num_layers", "num_prompts", "created_at", "cells", "summaries",
        ]
        return dict(zip(cols, row))

    def close(self) -> None:
        self._conn.close()
