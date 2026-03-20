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

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS commitment_runs (
                run_id TEXT PRIMARY KEY,
                dataset_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                n_prompts INTEGER NOT NULL,
                n_vulnerable INTEGER NOT NULL,
                n_robust INTEGER NOT NULL,
                threshold REAL NOT NULL,
                avg_commitment_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute(
            "CREATE SEQUENCE IF NOT EXISTS commitment_seq START 1"
        )
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS commitment_results (
                id INTEGER DEFAULT nextval('commitment_seq'),
                run_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                answer TEXT NOT NULL,
                peak_prob REAL,
                peak_layer INTEGER,
                final_prob REAL,
                min_margin REAL,
                margin_at_final REAL,
                competitor_token TEXT,
                competitor_peak REAL,
                crossover_layer INTEGER,
                vulnerable BOOLEAN,
                trajectory JSON,
                margin_trajectory JSON,
                competitor_trajectory JSON
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS universal_probes (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                heatmap_run_ids JSON NOT NULL,
                layer_range TEXT NOT NULL,
                n_samples INTEGER NOT NULL,
                n_vulnerable INTEGER NOT NULL,
                n_robust INTEGER NOT NULL,
                auc_roc REAL,
                cohens_d REAL,
                accuracy REAL,
                per_domain_auc JSON,
                model_path TEXT
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS contrast_runs (
                run_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                domains JSON NOT NULL,
                layers JSON NOT NULL,
                model_name TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS contrast_results (
                run_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                layer INTEGER NOT NULL,
                prompt TEXT NOT NULL,
                answer TEXT NOT NULL,
                mlp_delta_norm REAL,
                answer_projection REAL,
                competitor_projection REAL,
                competitor_token TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS contrast_summaries (
                run_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                layer INTEGER NOT NULL,
                mean_delta_norm REAL,
                std_delta_norm REAL,
                mean_answer_proj REAL,
                mean_competitor_proj REAL,
                cosine_similarities JSON
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS attribution_runs (
                run_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                layer INTEGER NOT NULL,
                target_direction TEXT NOT NULL,
                method TEXT NOT NULL,
                model_name TEXT NOT NULL,
                dataset TEXT
            )
        """)
        self._conn.execute(
            "CREATE SEQUENCE IF NOT EXISTS attribution_seq START 1"
        )
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS attribution_results (
                id INTEGER DEFAULT nextval('attribution_seq'),
                run_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                token_position INTEGER NOT NULL,
                token_text TEXT NOT NULL,
                attribution_score REAL NOT NULL,
                target_token TEXT NOT NULL,
                target_token_id INTEGER NOT NULL
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS token_trace_runs (
                run_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                dataset TEXT,
                model_name TEXT NOT NULL,
                layers JSON NOT NULL,
                prompt_count INTEGER NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS attention_trace_runs (
                run_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                dataset TEXT,
                model_name TEXT NOT NULL,
                layers JSON NOT NULL,
                prompt_count INTEGER NOT NULL
            )
        """)
        self._conn.execute(
            "CREATE SEQUENCE IF NOT EXISTS attention_trace_seq START 1"
        )
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS attention_trace_results (
                id INTEGER DEFAULT nextval('attention_trace_seq'),
                run_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                layer INTEGER NOT NULL,
                head_idx INTEGER NOT NULL,
                answer_projection REAL NOT NULL,
                magnitude REAL NOT NULL
            )
        """)

        self._conn.execute(
            "CREATE SEQUENCE IF NOT EXISTS token_trace_seq START 1"
        )
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS token_trace_results (
                id INTEGER DEFAULT nextval('token_trace_seq'),
                run_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                layer INTEGER NOT NULL,
                token_position INTEGER NOT NULL,
                token_text TEXT NOT NULL,
                is_subject BOOLEAN NOT NULL,
                is_last BOOLEAN NOT NULL,
                answer_projection REAL NOT NULL,
                competitor_projection REAL,
                delta_magnitude REAL NOT NULL
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS diagnosis_runs (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                dataset TEXT,
                model_name TEXT NOT NULL,
                prompt_count INTEGER NOT NULL
            )
        """)
        self._conn.execute(
            "CREATE SEQUENCE IF NOT EXISTS diagnosis_seq START 1"
        )
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS diagnosis_results (
                id INTEGER DEFAULT nextval('diagnosis_seq'),
                run_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                answer TEXT NOT NULL,
                total_attention REAL,
                total_mlp REAL,
                top3_share REAL,
                circuit_type TEXT,
                signal_strength TEXT,
                vulnerability TEXT,
                confidence TEXT,
                repair_json TEXT
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS decompose_runs (
                run_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                dataset TEXT,
                model_name TEXT NOT NULL,
                prompt_count INTEGER NOT NULL
            )
        """)
        self._conn.execute(
            "CREATE SEQUENCE IF NOT EXISTS decompose_seq START 1"
        )
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS decompose_results (
                id INTEGER DEFAULT nextval('decompose_seq'),
                run_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                answer TEXT NOT NULL,
                competitor TEXT NOT NULL,
                answer_logit REAL,
                competitor_logit REAL,
                margin REAL,
                embedding_margin REAL,
                component_json TEXT,
                reconstruction_error REAL
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

    def save_circuit(
        self, circuit_id: str, result, probe_id: str | None = None,
    ) -> None:
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

    def list_heatmap_runs(self) -> list[dict]:
        """List all heatmap runs."""
        rows = self._conn.execute(
            "SELECT run_id, dataset_name, model_name, adapter_path,"
            " num_layers, num_prompts, created_at"
            " FROM heatmap_runs ORDER BY created_at DESC"
        ).fetchall()
        cols = [
            "run_id", "dataset_name", "model_name", "adapter_path",
            "num_layers", "num_prompts", "created_at",
        ]
        return [dict(zip(cols, r)) for r in rows]

    def get_latest_heatmap_run_id(self) -> str:
        """Return the run_id of the most recent heatmap run."""
        row = self._conn.execute(
            "SELECT run_id FROM heatmap_runs ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            raise ValueError("No heatmap runs in database")
        return row[0]

    def get_all_heatmap_runs(self) -> list[dict]:
        """Return all heatmap runs with full data (including cells JSON)."""
        rows = self._conn.execute(
            "SELECT * FROM heatmap_runs ORDER BY created_at DESC"
        ).fetchall()
        cols = [
            "run_id", "dataset_name", "model_name", "adapter_path",
            "num_layers", "num_prompts", "created_at", "cells", "summaries",
        ]
        return [dict(zip(cols, r)) for r in rows]

    def save_universal_probe(
        self, probe_id: str, result, output_path: str | None = None,
    ) -> None:
        """Save a universal probe result to the database."""
        import json as json_mod

        self._conn.execute(
            "INSERT INTO universal_probes VALUES"
            " (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                probe_id,
                json_mod.dumps(result.heatmap_run_ids),
                f"{result.layer_range[0]}-{result.layer_range[1]}",
                result.n_samples,
                result.n_vulnerable,
                result.n_robust,
                result.auc_roc,
                result.cohens_d,
                result.accuracy,
                json_mod.dumps(result.per_domain_auc),
                output_path,
            ],
        )

    def write_commitment_run(
        self,
        run_id: str,
        dataset_name: str,
        model_name: str,
        n_prompts: int,
        n_vulnerable: int,
        n_robust: int,
        threshold: float,
        avg_commitment_score: float,
    ) -> None:
        """Write a commitment run to the database."""
        self._conn.execute(
            "INSERT INTO commitment_runs VALUES"
            " (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            [
                run_id,
                dataset_name,
                model_name,
                n_prompts,
                n_vulnerable,
                n_robust,
                threshold,
                avg_commitment_score,
            ],
        )

    def write_commitment_result(
        self,
        run_id: str,
        prompt: str,
        answer: str,
        peak_prob: float,
        peak_layer: int,
        final_prob: float,
        min_margin: float,
        margin_at_final: float,
        competitor_token: str,
        competitor_peak: float,
        crossover_layer: int | None,
        vulnerable: bool,
        trajectory: str,
        margin_trajectory: str,
        competitor_trajectory: str,
    ) -> None:
        """Write a single commitment result to the database."""
        self._conn.execute(
            "INSERT INTO commitment_results"
            " (run_id, prompt, answer, peak_prob, peak_layer,"
            "  final_prob, min_margin, margin_at_final,"
            "  competitor_token, competitor_peak,"
            "  crossover_layer, vulnerable,"
            "  trajectory, margin_trajectory,"
            "  competitor_trajectory)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run_id,
                prompt,
                answer,
                peak_prob,
                peak_layer,
                final_prob,
                min_margin,
                margin_at_final,
                competitor_token,
                competitor_peak,
                crossover_layer,
                vulnerable,
                trajectory,
                margin_trajectory,
                competitor_trajectory,
            ],
        )

    def read_commitment_run(self, run_id: str) -> dict:
        """Read a commitment run from the database."""
        row = self._conn.execute(
            "SELECT * FROM commitment_runs WHERE run_id = ?", [run_id]
        ).fetchone()
        if row is None:
            raise ValueError(f"Commitment run not found: {run_id}")
        cols = [
            "run_id", "dataset_name", "model_name",
            "n_prompts", "n_vulnerable", "n_robust",
            "threshold", "avg_commitment_score", "created_at",
        ]
        return dict(zip(cols, row))

    def read_commitment_results(self, run_id: str) -> list[dict]:
        """Read commitment results for a run."""
        rows = self._conn.execute(
            "SELECT * FROM commitment_results WHERE run_id = ?"
            " ORDER BY id",
            [run_id],
        ).fetchall()
        cols = [
            "id", "run_id", "prompt", "answer",
            "peak_prob", "peak_layer", "final_prob",
            "min_margin", "margin_at_final",
            "competitor_token", "competitor_peak",
            "crossover_layer", "vulnerable",
            "trajectory", "margin_trajectory",
            "competitor_trajectory",
        ]
        return [dict(zip(cols, r)) for r in rows]

    def list_commitment_runs(self) -> list[dict]:
        """List all commitment runs."""
        rows = self._conn.execute(
            "SELECT run_id, dataset_name, model_name,"
            " n_prompts, n_vulnerable, n_robust,"
            " threshold, avg_commitment_score, created_at"
            " FROM commitment_runs ORDER BY created_at DESC"
        ).fetchall()
        cols = [
            "run_id", "dataset_name", "model_name",
            "n_prompts", "n_vulnerable", "n_robust",
            "threshold", "avg_commitment_score", "created_at",
        ]
        return [dict(zip(cols, r)) for r in rows]

    def write_contrast_run(
        self,
        run_id: str,
        domains: str,
        layers: str,
        model_name: str,
    ) -> None:
        """Write a contrast run to the database."""
        self._conn.execute(
            "INSERT INTO contrast_runs VALUES"
            " (?, CURRENT_TIMESTAMP, ?, ?, ?)",
            [run_id, domains, layers, model_name],
        )

    def write_contrast_result(
        self,
        run_id: str,
        domain: str,
        layer: int,
        prompt: str,
        answer: str,
        mlp_delta_norm: float,
        answer_projection: float,
        competitor_projection: float,
        competitor_token: str,
    ) -> None:
        """Write a single contrast result to the database."""
        self._conn.execute(
            "INSERT INTO contrast_results VALUES"
            " (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run_id, domain, layer, prompt, answer,
                mlp_delta_norm, answer_projection,
                competitor_projection, competitor_token,
            ],
        )

    def write_contrast_summary(
        self,
        run_id: str,
        domain: str,
        layer: int,
        mean_delta_norm: float,
        std_delta_norm: float,
        mean_answer_proj: float,
        mean_competitor_proj: float,
        cosine_similarities: str,
    ) -> None:
        """Write a contrast domain summary to the database."""
        self._conn.execute(
            "INSERT INTO contrast_summaries VALUES"
            " (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run_id, domain, layer,
                mean_delta_norm, std_delta_norm,
                mean_answer_proj, mean_competitor_proj,
                cosine_similarities,
            ],
        )

    def read_contrast_run(self, run_id: str) -> dict:
        """Read a contrast run from the database."""
        row = self._conn.execute(
            "SELECT * FROM contrast_runs WHERE run_id = ?", [run_id]
        ).fetchone()
        if row is None:
            raise ValueError(f"Contrast run not found: {run_id}")
        cols = ["run_id", "created_at", "domains", "layers", "model_name"]
        return dict(zip(cols, row))

    def list_contrast_runs(self) -> list[dict]:
        """List all contrast runs."""
        rows = self._conn.execute(
            "SELECT run_id, created_at, domains, layers, model_name"
            " FROM contrast_runs ORDER BY created_at DESC"
        ).fetchall()
        cols = ["run_id", "created_at", "domains", "layers", "model_name"]
        return [dict(zip(cols, r)) for r in rows]

    def write_attribution_run(
        self,
        run_id: str,
        layer: int,
        target_direction: str,
        method: str,
        model_name: str,
        dataset: str | None = None,
    ) -> None:
        """Write an attribution run to the database."""
        self._conn.execute(
            "INSERT INTO attribution_runs VALUES"
            " (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)",
            [run_id, layer, target_direction, method, model_name, dataset],
        )

    def write_attribution_result(
        self,
        run_id: str,
        prompt: str,
        token_position: int,
        token_text: str,
        attribution_score: float,
        target_token: str,
        target_token_id: int,
    ) -> None:
        """Write a single attribution result to the database."""
        self._conn.execute(
            "INSERT INTO attribution_results"
            " (run_id, prompt, token_position, token_text,"
            "  attribution_score, target_token, target_token_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                run_id, prompt, token_position, token_text,
                attribution_score, target_token, target_token_id,
            ],
        )

    def read_attribution_run(self, run_id: str) -> dict:
        """Read an attribution run from the database."""
        row = self._conn.execute(
            "SELECT * FROM attribution_runs WHERE run_id = ?", [run_id]
        ).fetchone()
        if row is None:
            raise ValueError(f"Attribution run not found: {run_id}")
        cols = [
            "run_id", "created_at", "layer", "target_direction",
            "method", "model_name", "dataset",
        ]
        return dict(zip(cols, row))

    def read_attribution_results(self, run_id: str) -> list[dict]:
        """Read attribution results for a run."""
        rows = self._conn.execute(
            "SELECT * FROM attribution_results WHERE run_id = ?"
            " ORDER BY id",
            [run_id],
        ).fetchall()
        cols = [
            "id", "run_id", "prompt", "token_position",
            "token_text", "attribution_score",
            "target_token", "target_token_id",
        ]
        return [dict(zip(cols, r)) for r in rows]

    def list_attribution_runs(self) -> list[dict]:
        """List all attribution runs."""
        rows = self._conn.execute(
            "SELECT run_id, created_at, layer, target_direction,"
            " method, model_name, dataset"
            " FROM attribution_runs ORDER BY created_at DESC"
        ).fetchall()
        cols = [
            "run_id", "created_at", "layer", "target_direction",
            "method", "model_name", "dataset",
        ]
        return [dict(zip(cols, r)) for r in rows]

    def write_token_trace_run(
        self,
        run_id: str,
        dataset: str | None,
        model_name: str,
        layers: str,
        prompt_count: int,
    ) -> None:
        """Write a token-trace run to the database."""
        self._conn.execute(
            "INSERT INTO token_trace_runs VALUES"
            " (?, CURRENT_TIMESTAMP, ?, ?, ?, ?)",
            [run_id, dataset, model_name, layers, prompt_count],
        )

    def write_token_trace_result(
        self,
        run_id: str,
        prompt: str,
        layer: int,
        token_position: int,
        token_text: str,
        is_subject: bool,
        is_last: bool,
        answer_projection: float,
        competitor_projection: float | None,
        delta_magnitude: float,
    ) -> None:
        """Write a single token-trace result to the database."""
        self._conn.execute(
            "INSERT INTO token_trace_results"
            " (run_id, prompt, layer, token_position, token_text,"
            "  is_subject, is_last, answer_projection,"
            "  competitor_projection, delta_magnitude)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run_id, prompt, layer, token_position, token_text,
                is_subject, is_last, answer_projection,
                competitor_projection, delta_magnitude,
            ],
        )

    def list_token_trace_runs(self) -> list[dict]:
        """List all token-trace runs."""
        rows = self._conn.execute(
            "SELECT run_id, created_at, dataset, model_name,"
            " layers, prompt_count"
            " FROM token_trace_runs ORDER BY created_at DESC"
        ).fetchall()
        cols = [
            "run_id", "created_at", "dataset", "model_name",
            "layers", "prompt_count",
        ]
        return [dict(zip(cols, r)) for r in rows]

    def write_attention_trace_run(
        self,
        run_id: str,
        dataset: str | None,
        model_name: str,
        layers: str,
        prompt_count: int,
    ) -> None:
        """Write an attention-trace run to the database."""
        self._conn.execute(
            "INSERT INTO attention_trace_runs VALUES"
            " (?, CURRENT_TIMESTAMP, ?, ?, ?, ?)",
            [run_id, dataset, model_name, layers, prompt_count],
        )

    def write_attention_trace_result(
        self,
        run_id: str,
        prompt: str,
        layer: int,
        head_idx: int,
        answer_projection: float,
        magnitude: float,
    ) -> None:
        """Write a single attention-trace result to the database."""
        self._conn.execute(
            "INSERT INTO attention_trace_results"
            " (run_id, prompt, layer, head_idx,"
            "  answer_projection, magnitude)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [run_id, prompt, layer, head_idx, answer_projection, magnitude],
        )

    def list_attention_trace_runs(self) -> list[dict]:
        """List all attention-trace runs."""
        rows = self._conn.execute(
            "SELECT run_id, created_at, dataset, model_name,"
            " layers, prompt_count"
            " FROM attention_trace_runs ORDER BY created_at DESC"
        ).fetchall()
        cols = [
            "run_id", "created_at", "dataset", "model_name",
            "layers", "prompt_count",
        ]
        return [dict(zip(cols, r)) for r in rows]

    def write_diagnosis_run(
        self,
        run_id: str,
        dataset: str | None,
        model_name: str,
        prompt_count: int,
    ) -> None:
        """Write a diagnosis run to the database."""
        self._conn.execute(
            "INSERT INTO diagnosis_runs VALUES"
            " (?, CURRENT_TIMESTAMP, ?, ?, ?)",
            [run_id, dataset, model_name, prompt_count],
        )

    def write_diagnosis_result(
        self,
        run_id: str,
        prompt: str,
        answer: str,
        total_attention: float,
        total_mlp: float,
        top3_share: float,
        circuit_type: str,
        signal_strength: str,
        vulnerability: str,
        confidence: str,
        repair_json: str | None,
    ) -> None:
        """Write a single diagnosis result to the database."""
        self._conn.execute(
            "INSERT INTO diagnosis_results"
            " (run_id, prompt, answer, total_attention, total_mlp,"
            "  top3_share, circuit_type, signal_strength,"
            "  vulnerability, confidence, repair_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run_id, prompt, answer, total_attention, total_mlp,
                top3_share, circuit_type, signal_strength,
                vulnerability, confidence, repair_json,
            ],
        )

    def list_diagnosis_runs(self) -> list[dict]:
        """List all diagnosis runs."""
        rows = self._conn.execute(
            "SELECT id, created_at, dataset, model_name, prompt_count"
            " FROM diagnosis_runs ORDER BY created_at DESC"
        ).fetchall()
        cols = ["id", "created_at", "dataset", "model_name", "prompt_count"]
        return [dict(zip(cols, r)) for r in rows]

    def write_decompose_run(
        self,
        run_id: str,
        dataset: str | None,
        model_name: str,
        prompt_count: int,
    ) -> None:
        """Write a decompose run to the database."""
        self._conn.execute(
            "INSERT INTO decompose_runs VALUES"
            " (?, CURRENT_TIMESTAMP, ?, ?, ?)",
            [run_id, dataset, model_name, prompt_count],
        )

    def write_decompose_result(
        self,
        run_id: str,
        prompt: str,
        answer: str,
        competitor: str,
        answer_logit: float,
        competitor_logit: float,
        margin: float,
        embedding_margin: float,
        component_json: str,
        reconstruction_error: float,
    ) -> None:
        """Write a single decompose result to the database."""
        self._conn.execute(
            "INSERT INTO decompose_results"
            " (run_id, prompt, answer, competitor,"
            "  answer_logit, competitor_logit, margin,"
            "  embedding_margin, component_json,"
            "  reconstruction_error)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run_id, prompt, answer, competitor,
                answer_logit, competitor_logit, margin,
                embedding_margin, component_json,
                reconstruction_error,
            ],
        )

    def read_decompose_run(self, run_id: str) -> dict:
        """Read a decompose run from the database."""
        row = self._conn.execute(
            "SELECT * FROM decompose_runs WHERE run_id = ?", [run_id]
        ).fetchone()
        if row is None:
            raise ValueError(f"Decompose run not found: {run_id}")
        cols = [
            "run_id", "created_at", "dataset",
            "model_name", "prompt_count",
        ]
        return dict(zip(cols, row))

    def read_decompose_results(self, run_id: str) -> list[dict]:
        """Read decompose results for a run."""
        rows = self._conn.execute(
            "SELECT * FROM decompose_results WHERE run_id = ?"
            " ORDER BY id",
            [run_id],
        ).fetchall()
        cols = [
            "id", "run_id", "prompt", "answer", "competitor",
            "answer_logit", "competitor_logit", "margin",
            "embedding_margin", "component_json",
            "reconstruction_error",
        ]
        return [dict(zip(cols, r)) for r in rows]

    def list_decompose_runs(self) -> list[dict]:
        """List all decompose runs."""
        rows = self._conn.execute(
            "SELECT run_id, created_at, dataset, model_name, prompt_count"
            " FROM decompose_runs ORDER BY created_at DESC"
        ).fetchall()
        cols = [
            "run_id", "created_at", "dataset",
            "model_name", "prompt_count",
        ]
        return [dict(zip(cols, r)) for r in rows]

    def close(self) -> None:
        self._conn.close()
