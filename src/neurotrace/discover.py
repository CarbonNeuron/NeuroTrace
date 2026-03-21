"""Discover: automated knowledge gap detection via fact extraction."""

from __future__ import annotations

import html as _html
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Wikidata SPARQL queries and prompt templates
# ---------------------------------------------------------------------------

TOPIC_QUERIES: dict[str, str] = {
    "world_capitals": (
        "SELECT ?country ?countryLabel ?capital ?capitalLabel WHERE {{"
        " ?country wdt:P31 wd:Q6256."
        " ?country wdt:P36 ?capital."
        " SERVICE wikibase:label {{ bd:serviceParam wikibase:language \"en\". }}"
        " }} LIMIT {limit}"
    ),
    "chemical_elements": (
        "SELECT ?element ?elementLabel ?symbol WHERE {{"
        " ?element wdt:P31 wd:Q11344."
        " ?element wdt:P246 ?symbol."
        " SERVICE wikibase:label {{ bd:serviceParam wikibase:language \"en\". }}"
        " }} LIMIT {limit}"
    ),
    "country_currencies": (
        "SELECT ?country ?countryLabel ?currency ?currencyLabel WHERE {{"
        " ?country wdt:P31 wd:Q6256."
        " ?country wdt:P38 ?currency."
        " SERVICE wikibase:label {{ bd:serviceParam wikibase:language \"en\". }}"
        " }} LIMIT {limit}"
    ),
    "country_languages": (
        "SELECT ?country ?countryLabel ?language ?languageLabel WHERE {{"
        " ?country wdt:P31 wd:Q6256."
        " ?country wdt:P37 ?language."
        " SERVICE wikibase:label {{ bd:serviceParam wikibase:language \"en\". }}"
        " }} LIMIT {limit}"
    ),
    "planet_properties": (
        "SELECT ?planet ?planetLabel ?radius WHERE {{"
        " ?planet wdt:P31 wd:Q634."
        " ?planet wdt:P2120 ?radius."
        " SERVICE wikibase:label {{ bd:serviceParam wikibase:language \"en\". }}"
        " }} LIMIT {limit}"
    ),
}

TOPIC_TEMPLATES: dict[str, tuple[str, str]] = {
    "world_capitals": ("The capital of {countryLabel} is", "{capitalLabel}"),
    "chemical_elements": ("The chemical symbol for {elementLabel} is", "{symbol}"),
    "country_currencies": ("The currency of {countryLabel} is the", "{currencyLabel}"),
    "country_languages": (
        "The official language of {countryLabel} is",
        "{languageLabel}",
    ),
}

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "NeuroTrace/0.1 (https://github.com/CarbonNeuron/neurotrace)"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DiscoveryFact:
    """A single (prompt, answer) fact from a source."""

    id: str
    prompt: str
    expected_answer: str
    topic: str
    source: str
    baseline_correct: bool = False
    baseline_prob: float = 0.0
    healed: bool = False
    healed_prob: float = 0.0
    regression_flagged: bool = False


@dataclass
class DiscoveryResult:
    """Complete result of a discover run."""

    run_id: str
    topic: str
    source: str
    model_name: str
    total_facts: int
    baseline_correct: int
    baseline_wrong: int
    healed_count: int
    regression_count: int
    facts: list[DiscoveryFact] = field(default_factory=list)
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Wikidata client
# ---------------------------------------------------------------------------


def query_wikidata(
    sparql: str,
    max_retries: int = 3,
    initial_backoff: float = 1.0,
) -> list[dict]:
    """Execute a SPARQL query against Wikidata and return bindings.

    Implements exponential backoff for 429 responses.
    """
    import urllib.error
    import urllib.parse
    import urllib.request

    url = WIKIDATA_ENDPOINT + "?" + urllib.parse.urlencode({
        "query": sparql,
        "format": "json",
    })

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/sparql-results+json",
    }

    backoff = initial_backoff
    for attempt in range(max_retries):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["results"]["bindings"]
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise

    return []


def fetch_topic_facts(
    topic: str,
    limit: int = 100,
) -> list[dict]:
    """Fetch facts for a built-in topic from Wikidata.

    Returns list of {"prompt": ..., "answer": ...} dicts.
    """
    if topic not in TOPIC_QUERIES:
        available = ", ".join(TOPIC_QUERIES.keys())
        raise ValueError(f"Unknown topic: {topic!r}. Available: {available}")

    sparql = TOPIC_QUERIES[topic].format(limit=limit)
    bindings = query_wikidata(sparql)
    return expand_bindings(topic, bindings)


def expand_bindings(topic: str, bindings: list[dict]) -> list[dict]:
    """Expand SPARQL bindings into prompt/answer pairs using templates."""
    if topic not in TOPIC_TEMPLATES:
        raise ValueError(
            f"No template for topic {topic!r}. "
            f"Use --template and --answer-field for custom topics."
        )

    prompt_template, answer_template = TOPIC_TEMPLATES[topic]
    facts = []

    for binding in bindings:
        # Extract values from SPARQL binding format {"var": {"value": "..."}}
        values = {k: v["value"] for k, v in binding.items()}
        try:
            prompt = prompt_template.format(**values)
            answer = answer_template.format(**values)
            facts.append({"prompt": prompt, "answer": answer})
        except KeyError:
            continue

    return facts


def expand_custom_template(
    bindings: list[dict],
    template: str,
    answer_field: str,
) -> list[dict]:
    """Expand SPARQL bindings using a custom template and answer field."""
    facts = []
    for binding in bindings:
        values = {k: v["value"] for k, v in binding.items()}
        try:
            prompt = template.format(**values)
            answer = values[answer_field]
            facts.append({"prompt": prompt, "answer": answer})
        except KeyError:
            continue
    return facts


# ---------------------------------------------------------------------------
# File-based source
# ---------------------------------------------------------------------------


def load_facts_from_file(path: str) -> list[dict]:
    """Load facts from a JSONL file.

    Each line: {"prompt": "...", "answer": "..."}
    """
    facts = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "prompt" not in entry or "answer" not in entry:
                raise ValueError(
                    f"Line {line_num} missing 'prompt' or 'answer' field"
                )
            facts.append({"prompt": entry["prompt"], "answer": entry["answer"]})
    return facts


# ---------------------------------------------------------------------------
# Fingerprint regression checking
# ---------------------------------------------------------------------------


def check_regressions_fingerprint(
    edit_key,
    fingerprint_cache: dict[str, Any],
    similarity_threshold: float = 0.3,
) -> list[str]:
    """Return prompt IDs where the edit might cause regression.

    Uses cosine similarity between the edit's key vector and cached key vectors.
    """
    import torch

    flagged = []
    for prompt_id, cached_key in fingerprint_cache.items():
        similarity = torch.cosine_similarity(
            edit_key.unsqueeze(0), cached_key.unsqueeze(0), dim=1,
        )
        if similarity.item() > similarity_threshold:
            flagged.append(prompt_id)
    return flagged


def load_fingerprints(path: str) -> dict[str, Any]:
    """Load fingerprint cache from disk."""
    import torch

    return torch.load(path, weights_only=True)


def save_fingerprints(cache: dict[str, Any], path: str) -> None:
    """Save fingerprint cache to disk."""
    import torch

    torch.save(cache, path)


# ---------------------------------------------------------------------------
# DuckDB schema
# ---------------------------------------------------------------------------


def ensure_discoveries_table(db_path: str):
    """Create the discoveries table if it doesn't exist."""
    import duckdb

    con = duckdb.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS discoveries (
            id TEXT PRIMARY KEY,
            topic TEXT,
            source TEXT,
            prompt TEXT,
            expected_answer TEXT,
            baseline_correct BOOLEAN,
            baseline_prob FLOAT,
            healed BOOLEAN DEFAULT FALSE,
            healed_prob FLOAT,
            regression_flagged BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT now()
        )
    """)
    con.close()


def insert_discovery(db_path: str, fact: DiscoveryFact) -> None:
    """Insert or update a discovery fact in the database."""
    import duckdb

    con = duckdb.connect(db_path)
    con.execute(
        """
        INSERT OR REPLACE INTO discoveries
            (id, topic, source, prompt, expected_answer,
             baseline_correct, baseline_prob, healed, healed_prob,
             regression_flagged)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            fact.id,
            fact.topic,
            fact.source,
            fact.prompt,
            fact.expected_answer,
            fact.baseline_correct,
            fact.baseline_prob,
            fact.healed,
            fact.healed_prob,
            fact.regression_flagged,
        ],
    )
    con.close()


def get_cached_facts(db_path: str, topic: str) -> list[dict]:
    """Get previously cached SPARQL results from the database."""
    import duckdb

    con = duckdb.connect(db_path)
    try:
        rows = con.execute(
            "SELECT prompt, expected_answer FROM discoveries WHERE topic = ?",
            [topic],
        ).fetchall()
    except duckdb.CatalogException:
        return []
    finally:
        con.close()
    return [{"prompt": r[0], "answer": r[1]} for r in rows]


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def classify_prediction(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    seed: int = 42,
) -> tuple[bool, float]:
    """Run inference and classify whether the model gets the answer correct.

    Returns (is_correct, answer_prob).
    """
    import torch

    torch.manual_seed(seed)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not answer_ids:
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    answer_id = answer_ids[0] if answer_ids else 0

    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits.float(), dim=-1)
    answer_prob = probs[answer_id].item()
    top1_id = int(probs.argmax().item())

    # Check if top-1 matches answer token or starts with answer text
    is_correct = top1_id == answer_id
    if not is_correct:
        top1_text = tokenizer.decode(top1_id).strip().lstrip("\u2581").lower()
        answer_lower = answer.strip().lower()
        if top1_text and answer_lower.startswith(top1_text):
            is_correct = True
            answer_prob = probs[top1_id].item()

    return is_correct, answer_prob


# ---------------------------------------------------------------------------
# Core discovery pipeline
# ---------------------------------------------------------------------------


def run_discover(
    model,
    tokenizer,
    facts: list[dict],
    topic: str,
    source: str,
    db_path: str,
    heal: bool = False,
    fingerprint_path: str | None = None,
    regression_threshold: float = 0.3,
    max_edits: int = 50,
    dry_run: bool = False,
    save_path: str | None = None,
    seed: int = 42,
    progress_callback=None,
) -> DiscoveryResult:
    """Run the full discovery pipeline.

    1. Classify all facts (baseline scan)
    2. Optionally heal failures via ROME
    3. Optionally check regressions via fingerprints
    4. Store results in DuckDB
    """
    start_time = time.time()
    run_id = str(uuid.uuid4())[:8]
    model_name = model.config._name_or_path

    ensure_discoveries_table(db_path)

    # Step 1: Baseline scan
    discovery_facts: list[DiscoveryFact] = []
    baseline_correct = 0
    baseline_wrong = 0

    for i, entry in enumerate(facts):
        if progress_callback:
            progress_callback(
                "scan",
                f"Scanning {i + 1}/{len(facts)}: "
                f"{entry['prompt'][:40]}",
            )

        is_correct, prob = classify_prediction(
            model, tokenizer, entry["prompt"], entry["answer"], seed,
        )

        fact = DiscoveryFact(
            id=str(uuid.uuid4())[:8],
            prompt=entry["prompt"],
            expected_answer=entry["answer"],
            topic=topic,
            source=source,
            baseline_correct=is_correct,
            baseline_prob=prob,
        )
        discovery_facts.append(fact)

        if is_correct:
            baseline_correct += 1
        else:
            baseline_wrong += 1

    # Step 2: Heal failures if requested
    healed_count = 0
    regression_count = 0

    if heal and not dry_run:
        from neurotrace.repair import (
            compute_key_vector,
            get_answer_prob,
        )

        # Load or create fingerprint cache
        fp_cache: dict[str, Any] = {}
        if fingerprint_path:
            try:
                fp_cache = load_fingerprints(fingerprint_path)
            except (FileNotFoundError, Exception):
                fp_cache = {}

        failures = [f for f in discovery_facts if not f.baseline_correct]
        edits_done = 0

        for i, fact in enumerate(failures):
            if edits_done >= max_edits:
                break

            if progress_callback:
                progress_callback(
                    "heal",
                    f"Healing {i + 1}/{len(failures)}: {fact.prompt[:40]}",
                )

            try:
                from neurotrace.repair import run_repair_local

                result = run_repair_local(
                    model, tokenizer,
                    fact.prompt, fact.expected_answer,
                    seed=seed,
                )

                if result.status == "skipped":
                    continue

                # Check fingerprint regressions
                if fp_cache and fingerprint_path:
                    k_star = compute_key_vector(
                        model, tokenizer, fact.prompt,
                        result.target_layer, seed,
                    )
                    flagged = check_regressions_fingerprint(
                        k_star, fp_cache, regression_threshold,
                    )
                    if flagged:
                        fact.regression_flagged = True
                        regression_count += len(flagged)

                # Check if healed
                new_prob = get_answer_prob(
                    model, tokenizer, fact.prompt, fact.expected_answer, seed,
                )
                fact.healed = True
                fact.healed_prob = new_prob
                healed_count += 1
                edits_done += 1

                # Update fingerprint cache
                if fingerprint_path:
                    k_star = compute_key_vector(
                        model, tokenizer, fact.prompt,
                        result.target_layer, seed,
                    )
                    fp_cache[fact.id] = k_star.detach().cpu()

            except Exception:
                continue

        # Save fingerprint cache
        if fingerprint_path and fp_cache:
            save_fingerprints(fp_cache, fingerprint_path)

        # Save model if requested
        if save_path and edits_done > 0:
            import os

            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    # Step 3: Store in DuckDB
    for fact in discovery_facts:
        insert_discovery(db_path, fact)

    duration = time.time() - start_time

    return DiscoveryResult(
        run_id=run_id,
        topic=topic,
        source=source,
        model_name=model_name,
        total_facts=len(facts),
        baseline_correct=baseline_correct,
        baseline_wrong=baseline_wrong,
        healed_count=healed_count,
        regression_count=regression_count,
        facts=discovery_facts,
        duration_seconds=duration,
    )


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def discovery_result_to_dict(result: DiscoveryResult) -> dict:
    """Convert DiscoveryResult to JSON-serializable dict."""
    return {
        "run_id": result.run_id,
        "topic": result.topic,
        "source": result.source,
        "model": result.model_name,
        "total_facts": result.total_facts,
        "baseline": {
            "correct": result.baseline_correct,
            "wrong": result.baseline_wrong,
            "accuracy": (
                result.baseline_correct / result.total_facts
                if result.total_facts > 0
                else 0.0
            ),
        },
        "healed_count": result.healed_count,
        "regression_count": result.regression_count,
        "duration_seconds": result.duration_seconds,
        "facts": [
            {
                "id": f.id,
                "prompt": f.prompt,
                "expected_answer": f.expected_answer,
                "baseline_correct": f.baseline_correct,
                "baseline_prob": f.baseline_prob,
                "healed": f.healed,
                "healed_prob": f.healed_prob,
                "regression_flagged": f.regression_flagged,
            }
            for f in result.facts
        ],
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_BG = "#1a1a2e"
_BG2 = "#16213e"
_BG3 = "#0f3460"
_TEXT = "#e0e0e0"
_DIM = "#8a8a9a"
_ACCENT = "#e8956a"
_BLUE = "#64b5f6"
_GREEN = "#4caf50"
_RED = "#f44336"
_YELLOW = "#ffca28"


def _esc(s: Any) -> str:
    return _html.escape(str(s))


def _discover_css() -> str:
    return f"""
    :root {{
        --bg: {_BG}; --bg2: {_BG2}; --bg3: {_BG3};
        --text: {_TEXT}; --dim: {_DIM};
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        background: var(--bg); color: var(--text);
        font-family: -apple-system, BlinkMacSystemFont,
            'Segoe UI', Helvetica, Arial, sans-serif;
        font-size: 14px; line-height: 1.6; padding: 2rem;
        max-width: 1600px; margin: 0 auto;
    }}
    h1 {{ color: {_ACCENT}; font-size: 1.8rem; margin-bottom: 0.5rem; }}
    h2 {{
        color: {_BLUE}; font-size: 1.3rem; margin: 2rem 0 1rem;
        border-bottom: 1px solid var(--bg3); padding-bottom: 0.5rem;
    }}
    .meta {{
        background: var(--bg2); padding: 1.5rem;
        border-radius: 8px; margin-bottom: 2rem;
    }}
    .meta-grid {{
        display: grid; gap: 0.5rem 2rem;
        grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    }}
    .meta-label {{ color: var(--dim); font-size: 0.85rem; }}
    .meta-value {{
        color: var(--text);
        font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    }}
    .cards {{
        display: grid; gap: 1rem;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        margin-bottom: 2rem;
    }}
    .card {{
        background: var(--bg2); padding: 1.2rem;
        border-radius: 8px; text-align: center;
    }}
    .card-value {{
        font-size: 2rem; font-weight: bold;
        font-family: 'SF Mono', 'Fira Code', monospace;
    }}
    .card-label {{ color: var(--dim); font-size: 0.85rem; margin-top: 0.25rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th {{
        background: var(--bg3); color: var(--dim); text-align: left;
        padding: 0.5rem 0.75rem; font-weight: 600; font-size: 0.8rem;
        text-transform: uppercase; letter-spacing: 0.05em;
    }}
    td {{
        padding: 0.4rem 0.75rem;
        border-bottom: 1px solid var(--bg3);
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 0.85rem;
    }}
    .correct {{ color: {_GREEN}; font-weight: bold; }}
    .wrong {{ color: {_RED}; font-weight: bold; }}
    .healed {{ color: {_YELLOW}; font-weight: bold; }}
    @media print {{
        body {{ background: white; color: #222; padding: 1rem; }}
        .meta, .card {{ background: #f5f5f5; color: #222; }}
        th {{ background: #e0e0e0; color: #333; }}
        td {{ border-color: #ccc; color: #222; }}
        h1 {{ color: #333; }} h2 {{ color: #555; }}
    }}
    """


def generate_discover_html(result: DiscoveryResult) -> str:
    """Generate self-contained HTML report for a discover run."""
    baseline_acc = (
        result.baseline_correct / result.total_facts
        if result.total_facts > 0
        else 0.0
    )
    healed_total = result.baseline_correct + result.healed_count
    healed_acc = healed_total / result.total_facts if result.total_facts > 0 else 0.0

    parts = []

    # Header
    parts.append(f"""
    <h1>NeuroTrace Discover Report</h1>
    <div class="meta">
        <div class="meta-grid">
            <div><span class="meta-label">Topic</span><br>
                <span class="meta-value">{_esc(result.topic)}</span></div>
            <div><span class="meta-label">Source</span><br>
                <span class="meta-value">{_esc(result.source)}</span></div>
            <div><span class="meta-label">Model</span><br>
                <span class="meta-value">{_esc(result.model_name)}</span></div>
            <div><span class="meta-label">Duration</span><br>
                <span class="meta-value">{result.duration_seconds:.1f}s</span></div>
        </div>
    </div>
    """)

    # Summary cards
    acc_color = _GREEN if healed_acc > baseline_acc else _BLUE
    parts.append(f"""
    <div class="cards">
        <div class="card">
            <div class="card-value">{result.total_facts}</div>
            <div class="card-label">Total Facts</div>
        </div>
        <div class="card">
            <div class="card-value">{baseline_acc:.0%}</div>
            <div class="card-label">Baseline Accuracy</div>
        </div>
        <div class="card">
            <div class="card-value" style="color:{acc_color}">{healed_acc:.0%}</div>
            <div class="card-label">After Healing</div>
        </div>
        <div class="card">
            <div class="card-value">{result.healed_count}</div>
            <div class="card-label">Healed</div>
        </div>
        <div class="card">
            <div class="card-value">{result.regression_count}</div>
            <div class="card-label">Regressions</div>
        </div>
    </div>
    """)

    # Facts table
    parts.append("<h2>Facts</h2>")
    parts.append(
        "<table><thead><tr>"
        "<th>Prompt</th><th>Expected</th><th>Baseline</th>"
        "<th>Prob</th><th>Healed</th><th>Healed Prob</th>"
        "</tr></thead><tbody>"
    )
    for f in result.facts:
        status_class = "correct" if f.baseline_correct else "wrong"
        heal_class = "healed" if f.healed else ""
        parts.append(
            f"<tr>"
            f"<td>{_esc(f.prompt[:60])}</td>"
            f"<td>{_esc(f.expected_answer)}</td>"
            f'<td class="{status_class}">'
            f'{"CORRECT" if f.baseline_correct else "WRONG"}</td>'
            f"<td>{f.baseline_prob:.2%}</td>"
            f'<td class="{heal_class}">'
            f'{"YES" if f.healed else "-"}</td>'
            f"<td>{f'{f.healed_prob:.2%}' if f.healed else '-'}</td>"
            f"</tr>"
        )
    parts.append("</tbody></table>")

    body = "\n".join(parts)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Discover Report - {_esc(result.topic)}</title>\n"
        f"<style>{_discover_css()}</style>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>"
    )
