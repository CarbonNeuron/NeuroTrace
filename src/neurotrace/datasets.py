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
    {"prompt": "The capital of India is", "answer": "New Delhi"},
    {"prompt": "The capital of Brazil is", "answer": "Brasilia"},
    {"prompt": "The capital of Canada is", "answer": "Ottawa"},
    {"prompt": "The capital of Australia is", "answer": "Canberra"},
    {"prompt": "The capital of South Korea is", "answer": "Seoul"},
    {"prompt": "The capital of Mexico is", "answer": "Mexico City"},
    {"prompt": "The capital of Argentina is", "answer": "Buenos Aires"},
    {"prompt": "The capital of Egypt is", "answer": "Cairo"},
    {"prompt": "The capital of Turkey is", "answer": "Ankara"},
    {"prompt": "The capital of Thailand is", "answer": "Bangkok"},
    {"prompt": "The capital of Indonesia is", "answer": "Jakarta"},
    {"prompt": "The capital of Poland is", "answer": "Warsaw"},
    {"prompt": "The capital of Sweden is", "answer": "Stockholm"},
    {"prompt": "The capital of Norway is", "answer": "Oslo"},
    {"prompt": "The capital of Denmark is", "answer": "Copenhagen"},
    {"prompt": "The capital of Finland is", "answer": "Helsinki"},
    {"prompt": "The capital of Greece is", "answer": "Athens"},
    {"prompt": "The capital of Portugal is", "answer": "Lisbon"},
    {"prompt": "The capital of Austria is", "answer": "Vienna"},
    {"prompt": "The capital of Switzerland is", "answer": "Bern"},
    {"prompt": "The capital of Netherlands is", "answer": "Amsterdam"},
    {"prompt": "The capital of Belgium is", "answer": "Brussels"},
    {"prompt": "The capital of Ireland is", "answer": "Dublin"},
    {"prompt": "The capital of Ukraine is", "answer": "Kyiv"},
    {"prompt": "The capital of South Africa is", "answer": "Pretoria"},
    {"prompt": "The capital of Kenya is", "answer": "Nairobi"},
    {"prompt": "The capital of Nigeria is", "answer": "Abuja"},
    {"prompt": "The capital of Peru is", "answer": "Lima"},
    {"prompt": "The capital of Colombia is", "answer": "Bogota"},
    {"prompt": "The capital of Chile is", "answer": "Santiago"},
    {"prompt": "The capital of Vietnam is", "answer": "Hanoi"},
    {"prompt": "The capital of Philippines is", "answer": "Manila"},
    {"prompt": "The capital of Malaysia is", "answer": "Kuala Lumpur"},
    {"prompt": "The capital of New Zealand is", "answer": "Wellington"},
    {"prompt": "The capital of Czech Republic is", "answer": "Prague"},
    {"prompt": "The capital of Romania is", "answer": "Bucharest"},
    {"prompt": "The capital of Hungary is", "answer": "Budapest"},
    {"prompt": "The capital of Israel is", "answer": "Jerusalem"},
    {"prompt": "The capital of Saudi Arabia is", "answer": "Riyadh"},
    {"prompt": "The capital of Iran is", "answer": "Tehran"},
    {"prompt": "The capital of Pakistan is", "answer": "Islamabad"},
    {"prompt": "The capital of Myanmar is", "answer": "Naypyidaw"},
]

MATH_SIMPLE = [
    # Addition (15 examples)
    {"prompt": "2 + 3 =", "answer": "5"},
    {"prompt": "7 + 8 =", "answer": "15"},
    {"prompt": "4 + 9 =", "answer": "13"},
    {"prompt": "6 + 5 =", "answer": "11"},
    {"prompt": "3 + 8 =", "answer": "11"},
    {"prompt": "1 + 7 =", "answer": "8"},
    {"prompt": "9 + 2 =", "answer": "11"},
    {"prompt": "5 + 5 =", "answer": "10"},
    {"prompt": "8 + 4 =", "answer": "12"},
    {"prompt": "3 + 6 =", "answer": "9"},
    {"prompt": "7 + 7 =", "answer": "14"},
    {"prompt": "2 + 9 =", "answer": "11"},
    {"prompt": "6 + 8 =", "answer": "14"},
    {"prompt": "4 + 3 =", "answer": "7"},
    {"prompt": "1 + 1 =", "answer": "2"},
    # Subtraction (10 examples)
    {"prompt": "9 - 4 =", "answer": "5"},
    {"prompt": "15 - 7 =", "answer": "8"},
    {"prompt": "12 - 5 =", "answer": "7"},
    {"prompt": "8 - 3 =", "answer": "5"},
    {"prompt": "10 - 6 =", "answer": "4"},
    {"prompt": "7 - 2 =", "answer": "5"},
    {"prompt": "14 - 9 =", "answer": "5"},
    {"prompt": "11 - 8 =", "answer": "3"},
    {"prompt": "6 - 1 =", "answer": "5"},
    {"prompt": "13 - 6 =", "answer": "7"},
    # Multiplication (15 examples)
    {"prompt": "3 \u00d7 4 =", "answer": "12"},
    {"prompt": "7 \u00d7 6 =", "answer": "42"},
    {"prompt": "9 \u00d7 7 =", "answer": "63"},
    {"prompt": "8 \u00d7 6 =", "answer": "48"},
    {"prompt": "5 \u00d7 5 =", "answer": "25"},
    {"prompt": "2 \u00d7 8 =", "answer": "16"},
    {"prompt": "4 \u00d7 7 =", "answer": "28"},
    {"prompt": "6 \u00d7 3 =", "answer": "18"},
    {"prompt": "9 \u00d7 9 =", "answer": "81"},
    {"prompt": "8 \u00d7 7 =", "answer": "56"},
    {"prompt": "3 \u00d7 9 =", "answer": "27"},
    {"prompt": "5 \u00d7 6 =", "answer": "30"},
    {"prompt": "4 \u00d7 4 =", "answer": "16"},
    {"prompt": "7 \u00d7 8 =", "answer": "56"},
    {"prompt": "2 \u00d7 7 =", "answer": "14"},
    # Division (10 examples)
    {"prompt": "12 / 4 =", "answer": "3"},
    {"prompt": "45 / 9 =", "answer": "5"},
    {"prompt": "24 / 6 =", "answer": "4"},
    {"prompt": "36 / 9 =", "answer": "4"},
    {"prompt": "18 / 3 =", "answer": "6"},
    {"prompt": "21 / 7 =", "answer": "3"},
    {"prompt": "40 / 8 =", "answer": "5"},
    {"prompt": "15 / 5 =", "answer": "3"},
    {"prompt": "56 / 8 =", "answer": "7"},
    {"prompt": "27 / 9 =", "answer": "3"},
]

HISTORY_DATES = [
    # Wars and conflicts (10)
    {"prompt": "World War 2 ended in the year", "answer": "1945"},
    {"prompt": "World War 1 began in the year", "answer": "1914"},
    {"prompt": "The Korean War began in the year", "answer": "1950"},
    {"prompt": "The American Civil War ended in the year", "answer": "1865"},
    {"prompt": "The Battle of Waterloo took place in the year", "answer": "1815"},
    {"prompt": "The Vietnam War ended in the year", "answer": "1975"},
    {"prompt": "The Falklands War took place in the year", "answer": "1982"},
    {"prompt": "The Gulf War began in the year", "answer": "1990"},
    {"prompt": "The Hundred Years War ended in the year", "answer": "1453"},
    {"prompt": "The Russo-Japanese War began in the year", "answer": "1904"},
    # Political events (10)
    {"prompt": "The Declaration of Independence was signed in", "answer": "1776"},
    {"prompt": "The Berlin Wall fell in", "answer": "1989"},
    {"prompt": "The French Revolution began in", "answer": "1789"},
    {"prompt": "The Russian Revolution took place in", "answer": "1917"},
    {"prompt": "The Magna Carta was signed in", "answer": "1215"},
    {"prompt": "Nelson Mandela was released from prison in", "answer": "1990"},
    {"prompt": "The United Nations was founded in", "answer": "1945"},
    {"prompt": "The European Union was established in", "answer": "1993"},
    {"prompt": "India gained independence in the year", "answer": "1947"},
    {"prompt": "The reunification of Germany occurred in", "answer": "1990"},
    # Scientific milestones (10)
    {"prompt": "The first moon landing occurred in the year", "answer": "1969"},
    {"prompt": "Penicillin was discovered in the year", "answer": "1928"},
    {"prompt": "The first nuclear bomb was detonated in the year", "answer": "1945"},
    {"prompt": "The structure of DNA was discovered in the year", "answer": "1953"},
    {"prompt": "The first heart transplant was performed in", "answer": "1967"},
    {"prompt": "The World Wide Web was invented in the year", "answer": "1989"},
    {"prompt": "The Wright brothers first flew in", "answer": "1903"},
    {"prompt": "Einstein published general relativity in", "answer": "1915"},
    {"prompt": "The Hubble Space Telescope was launched in the year", "answer": "1990"},
    {"prompt": "Dolly the sheep was cloned in", "answer": "1996"},
    # Cultural/social milestones (10)
    {"prompt": "The Titanic sank in the year", "answer": "1912"},
    {"prompt": "Women gained the right to vote in the US in", "answer": "1920"},
    {"prompt": "The first modern Olympic Games were in", "answer": "1896"},
    {"prompt": "MLK gave the I Have a Dream speech in", "answer": "1963"},
    {"prompt": "Slavery was abolished in the US in", "answer": "1865"},
    {"prompt": "The first iPhone was released in the year", "answer": "2007"},
    {"prompt": "Gutenberg invented the printing press around", "answer": "1440"},
    {"prompt": "Shakespeare was born in the year", "answer": "1564"},
    {"prompt": "The Great Fire of London was in", "answer": "1666"},
    {"prompt": "The Panama Canal was completed in the year", "answer": "1914"},
]

SCIENCE_SYMBOLS = [
    # Chemical symbols (20)
    {"prompt": "The chemical symbol for gold is", "answer": "Au"},
    {"prompt": "The chemical symbol for iron is", "answer": "Fe"},
    {"prompt": "The chemical symbol for silver is", "answer": "Ag"},
    {"prompt": "The chemical symbol for copper is", "answer": "Cu"},
    {"prompt": "The chemical symbol for sodium is", "answer": "Na"},
    {"prompt": "The chemical symbol for potassium is", "answer": "K"},
    {"prompt": "The chemical symbol for mercury is", "answer": "Hg"},
    {"prompt": "The chemical symbol for lead is", "answer": "Pb"},
    {"prompt": "The chemical symbol for tin is", "answer": "Sn"},
    {"prompt": "The chemical symbol for tungsten is", "answer": "W"},
    {"prompt": "The chemical symbol for helium is", "answer": "He"},
    {"prompt": "The chemical symbol for neon is", "answer": "Ne"},
    {"prompt": "The chemical symbol for argon is", "answer": "Ar"},
    {"prompt": "The chemical symbol for oxygen is", "answer": "O"},
    {"prompt": "The chemical symbol for nitrogen is", "answer": "N"},
    {"prompt": "The chemical symbol for carbon is", "answer": "C"},
    {"prompt": "The chemical symbol for hydrogen is", "answer": "H"},
    {"prompt": "The chemical symbol for calcium is", "answer": "Ca"},
    {"prompt": "The chemical symbol for zinc is", "answer": "Zn"},
    {"prompt": "The chemical symbol for platinum is", "answer": "Pt"},
    # Physical constants (10)
    {"prompt": "The speed of light in m/s is approximately", "answer": "299"},
    {"prompt": "Absolute zero in Celsius is", "answer": "-273"},
    {"prompt": "Acceleration due to gravity in m/s2 is about", "answer": "9"},
    {"prompt": "The number of planets in our solar system is", "answer": "8"},
    {"prompt": "The charge of an electron in units of e is", "answer": "-1"},
    {"prompt": "The atomic number of carbon is", "answer": "6"},
    {"prompt": "The atomic number of oxygen is", "answer": "8"},
    {"prompt": "The atomic number of hydrogen is", "answer": "1"},
    {"prompt": "The atomic number of gold is", "answer": "79"},
    {"prompt": "The atomic number of iron is", "answer": "26"},
    # Basic science facts (10)
    {"prompt": "Water boils at", "answer": "100"},
    {"prompt": "Water freezes at", "answer": "0"},
    {"prompt": "The number of chromosomes in a human cell is", "answer": "46"},
    {"prompt": "The pH of pure water is", "answer": "7"},
    {"prompt": "The speed of sound in m/s at sea level is about", "answer": "343"},
    {"prompt": "The number of bones in an adult human body is", "answer": "206"},
    {"prompt": "The boiling point of nitrogen in Celsius is", "answer": "-196"},
    {"prompt": "The melting point of iron in Celsius is about", "answer": "1538"},
    {"prompt": "The density of water in kg/m3 is", "answer": "1000"},
    {"prompt": "Normal human body temperature in Fahrenheit is", "answer": "98"},
]

_BUILTIN_DATASETS = {
    "capitals": CAPITALS,
    "math_simple": MATH_SIMPLE,
    "history_dates": HISTORY_DATES,
    "science_symbols": SCIENCE_SYMBOLS,
}


def get_builtin_dataset(name: str) -> list[dict]:
    """Get a built-in dataset by name. Use 'all' for combined datasets."""
    if name == "all":
        combined = []
        for ds in _BUILTIN_DATASETS.values():
            combined.extend(ds)
        return combined
    if name not in _BUILTIN_DATASETS:
        available = ", ".join(list(_BUILTIN_DATASETS.keys()) + ["all"])
        raise ValueError(f"Unknown built-in dataset: {name!r}. Available: {available}")
    return _BUILTIN_DATASETS[name]


def list_builtin_datasets() -> list[str]:
    """Return list of available built-in dataset names."""
    return list(_BUILTIN_DATASETS.keys())


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
