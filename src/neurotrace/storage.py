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
            self._conn.execute(
                "ALTER TABLE traces ADD COLUMN interventions VARCHAR"
            )
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
                f"Ambiguous label {identifier!r}: matches "
                f"{len(rows)} traces"
            )

        # Try prefix match on trace_id
        rows = self._conn.execute(
            "SELECT trace_id FROM traces"
            " WHERE trace_id LIKE ? || '%'",
            [identifier],
        ).fetchall()
        if len(rows) == 1:
            return rows[0][0]
        if len(rows) > 1:
            raise ValueError(
                f"Ambiguous prefix {identifier!r}: matches "
                f"{len(rows)} traces"
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

    def close(self) -> None:
        self._conn.close()
