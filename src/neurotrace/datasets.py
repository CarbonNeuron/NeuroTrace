"""Built-in datasets for scan command."""

import json

CAPITALS = [
    {"prompt": "The capital of France is", "answer": "Paris"},
    {"prompt": "The capital of Germany is", "answer": "Berlin"},
    {"prompt": "The capital of Japan is", "answer": "Tokyo"},
    {"prompt": "The capital of Italy is", "answer": "Rome"},
    {"prompt": "The capital of Spain is", "answer": "Madrid"},
    {"prompt": "The capital of the United Kingdom is", "answer": "London"},
    {"prompt": "The capital of China is", "answer": "Beijing"},
    {"prompt": "The capital of Russia is", "answer": "Moscow"},
    {"prompt": "The capital of India is", "answer": "New"},
    {"prompt": "The capital of Brazil is", "answer": "Bras"},
    {"prompt": "The capital of Canada is", "answer": "Ottawa"},
    {"prompt": "The capital of Australia is", "answer": "Can"},
    {"prompt": "The capital of South Korea is", "answer": "Se"},
    {"prompt": "The capital of Mexico is", "answer": "Mexico"},
    {"prompt": "The capital of Argentina is", "answer": "Buenos"},
    {"prompt": "The capital of Egypt is", "answer": "Cairo"},
    {"prompt": "The capital of Turkey is", "answer": "Ank"},
    {"prompt": "The capital of Thailand is", "answer": "Bang"},
    {"prompt": "The capital of Indonesia is", "answer": "Jak"},
    {"prompt": "The capital of Poland is", "answer": "Warsaw"},
    {"prompt": "The capital of Sweden is", "answer": "Stock"},
    {"prompt": "The capital of Norway is", "answer": "Oslo"},
    {"prompt": "The capital of Denmark is", "answer": "Cop"},
    {"prompt": "The capital of Finland is", "answer": "Hels"},
    {"prompt": "The capital of Greece is", "answer": "Athens"},
    {"prompt": "The capital of Portugal is", "answer": "Lis"},
    {"prompt": "The capital of Austria is", "answer": "Vienna"},
    {"prompt": "The capital of Switzerland is", "answer": "Bern"},
    {"prompt": "The capital of Netherlands is", "answer": "Amsterdam"},
    {"prompt": "The capital of Belgium is", "answer": "Bru"},
    {"prompt": "The capital of Ireland is", "answer": "Dublin"},
    {"prompt": "The capital of Ukraine is", "answer": "Ky"},
    {"prompt": "The capital of South Africa is", "answer": "Pret"},
    {"prompt": "The capital of Kenya is", "answer": "Na"},
    {"prompt": "The capital of Nigeria is", "answer": "Ab"},
    {"prompt": "The capital of Peru is", "answer": "Lima"},
    {"prompt": "The capital of Colombia is", "answer": "Bog"},
    {"prompt": "The capital of Chile is", "answer": "Santiago"},
    {"prompt": "The capital of Vietnam is", "answer": "Han"},
    {"prompt": "The capital of Philippines is", "answer": "Man"},
    {"prompt": "The capital of Malaysia is", "answer": "Ku"},
    {"prompt": "The capital of New Zealand is", "answer": "Wellington"},
    {"prompt": "The capital of Czech Republic is", "answer": "Pr"},
    {"prompt": "The capital of Romania is", "answer": "Buch"},
    {"prompt": "The capital of Hungary is", "answer": "Bud"},
    {"prompt": "The capital of Israel is", "answer": "Jer"},
    {"prompt": "The capital of Saudi Arabia is", "answer": "R"},
    {"prompt": "The capital of Iran is", "answer": "Teh"},
    {"prompt": "The capital of Pakistan is", "answer": "Islam"},
    {"prompt": "The capital of Myanmar is", "answer": "Nay"},
]

_BUILTIN_DATASETS = {
    "capitals": CAPITALS,
}


def get_builtin_dataset(name: str) -> list[dict]:
    """Get a built-in dataset by name."""
    if name not in _BUILTIN_DATASETS:
        available = ", ".join(_BUILTIN_DATASETS.keys())
        raise ValueError(f"Unknown built-in dataset: {name!r}. Available: {available}")
    return _BUILTIN_DATASETS[name]


def load_dataset(path: str) -> list[dict]:
    """Load a dataset from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array")
    for i, entry in enumerate(data):
        missing = [k for k in ("prompt", "answer") if k not in entry]
        if missing:
            raise ValueError(f"Entry {i} missing required fields: {', '.join(missing)}")
    return data
