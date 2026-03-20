"""Built-in datasets for scan command."""

import json

CAPITALS = [
    {"prompt": "The capital of France is", "answer": "Paris", "subject": "France"},
    {"prompt": "The capital of Germany is", "answer": "Berlin", "subject": "Germany"},
    {"prompt": "The capital of Japan is", "answer": "Tokyo", "subject": "Japan"},
    {"prompt": "The capital of Italy is", "answer": "Rome", "subject": "Italy"},
    {"prompt": "The capital of Spain is", "answer": "Madrid", "subject": "Spain"},
    {
        "prompt": "The capital of the United Kingdom is",
        "answer": "London",
        "subject": "the United Kingdom",
    },
    {"prompt": "The capital of China is", "answer": "Beijing", "subject": "China"},
    {"prompt": "The capital of Russia is", "answer": "Moscow", "subject": "Russia"},
    {"prompt": "The capital of India is", "answer": "New Delhi", "subject": "India"},
    {"prompt": "The capital of Brazil is", "answer": "Brasilia", "subject": "Brazil"},
    {"prompt": "The capital of Canada is", "answer": "Ottawa", "subject": "Canada"},
    {
        "prompt": "The capital of Australia is",
        "answer": "Canberra",
        "subject": "Australia",
    },
    {
        "prompt": "The capital of South Korea is",
        "answer": "Seoul",
        "subject": "South Korea",
    },
    {
        "prompt": "The capital of Mexico is",
        "answer": "Mexico City",
        "subject": "Mexico",
    },
    {
        "prompt": "The capital of Argentina is",
        "answer": "Buenos Aires",
        "subject": "Argentina",
    },
    {"prompt": "The capital of Egypt is", "answer": "Cairo", "subject": "Egypt"},
    {"prompt": "The capital of Turkey is", "answer": "Ankara", "subject": "Turkey"},
    {
        "prompt": "The capital of Thailand is",
        "answer": "Bangkok",
        "subject": "Thailand",
    },
    {
        "prompt": "The capital of Indonesia is",
        "answer": "Jakarta",
        "subject": "Indonesia",
    },
    {"prompt": "The capital of Poland is", "answer": "Warsaw", "subject": "Poland"},
    {"prompt": "The capital of Sweden is", "answer": "Stockholm", "subject": "Sweden"},
    {"prompt": "The capital of Norway is", "answer": "Oslo", "subject": "Norway"},
    {
        "prompt": "The capital of Denmark is",
        "answer": "Copenhagen",
        "subject": "Denmark",
    },
    {"prompt": "The capital of Finland is", "answer": "Helsinki", "subject": "Finland"},
    {"prompt": "The capital of Greece is", "answer": "Athens", "subject": "Greece"},
    {"prompt": "The capital of Portugal is", "answer": "Lisbon", "subject": "Portugal"},
    {"prompt": "The capital of Austria is", "answer": "Vienna", "subject": "Austria"},
    {
        "prompt": "The capital of Switzerland is",
        "answer": "Bern",
        "subject": "Switzerland",
    },
    {
        "prompt": "The capital of Netherlands is",
        "answer": "Amsterdam",
        "subject": "Netherlands",
    },
    {"prompt": "The capital of Belgium is", "answer": "Brussels", "subject": "Belgium"},
    {"prompt": "The capital of Ireland is", "answer": "Dublin", "subject": "Ireland"},
    {"prompt": "The capital of Ukraine is", "answer": "Kyiv", "subject": "Ukraine"},
    {
        "prompt": "The capital of South Africa is",
        "answer": "Pretoria",
        "subject": "South Africa",
    },
    {"prompt": "The capital of Kenya is", "answer": "Nairobi", "subject": "Kenya"},
    {"prompt": "The capital of Nigeria is", "answer": "Abuja", "subject": "Nigeria"},
    {"prompt": "The capital of Peru is", "answer": "Lima", "subject": "Peru"},
    {"prompt": "The capital of Colombia is", "answer": "Bogota", "subject": "Colombia"},
    {"prompt": "The capital of Chile is", "answer": "Santiago", "subject": "Chile"},
    {"prompt": "The capital of Vietnam is", "answer": "Hanoi", "subject": "Vietnam"},
    {
        "prompt": "The capital of Philippines is",
        "answer": "Manila",
        "subject": "Philippines",
    },
    {
        "prompt": "The capital of Malaysia is",
        "answer": "Kuala Lumpur",
        "subject": "Malaysia",
    },
    {
        "prompt": "The capital of New Zealand is",
        "answer": "Wellington",
        "subject": "New Zealand",
    },
    {
        "prompt": "The capital of Czech Republic is",
        "answer": "Prague",
        "subject": "Czech Republic",
    },
    {
        "prompt": "The capital of Romania is",
        "answer": "Bucharest",
        "subject": "Romania",
    },
    {"prompt": "The capital of Hungary is", "answer": "Budapest", "subject": "Hungary"},
    {"prompt": "The capital of Israel is", "answer": "Jerusalem", "subject": "Israel"},
    {
        "prompt": "The capital of Saudi Arabia is",
        "answer": "Riyadh",
        "subject": "Saudi Arabia",
    },
    {"prompt": "The capital of Iran is", "answer": "Tehran", "subject": "Iran"},
    {
        "prompt": "The capital of Pakistan is",
        "answer": "Islamabad",
        "subject": "Pakistan",
    },
    {
        "prompt": "The capital of Myanmar is",
        "answer": "Naypyidaw",
        "subject": "Myanmar",
    },
]

MATH_SIMPLE = [
    # Addition (15 examples)
    {"prompt": "2 + 3 =", "answer": "5", "subject": "2 + 3"},
    {"prompt": "7 + 8 =", "answer": "15", "subject": "7 + 8"},
    {"prompt": "4 + 9 =", "answer": "13", "subject": "4 + 9"},
    {"prompt": "6 + 5 =", "answer": "11", "subject": "6 + 5"},
    {"prompt": "3 + 8 =", "answer": "11", "subject": "3 + 8"},
    {"prompt": "1 + 7 =", "answer": "8", "subject": "1 + 7"},
    {"prompt": "9 + 2 =", "answer": "11", "subject": "9 + 2"},
    {"prompt": "5 + 5 =", "answer": "10", "subject": "5 + 5"},
    {"prompt": "8 + 4 =", "answer": "12", "subject": "8 + 4"},
    {"prompt": "3 + 6 =", "answer": "9", "subject": "3 + 6"},
    {"prompt": "7 + 7 =", "answer": "14", "subject": "7 + 7"},
    {"prompt": "2 + 9 =", "answer": "11", "subject": "2 + 9"},
    {"prompt": "6 + 8 =", "answer": "14", "subject": "6 + 8"},
    {"prompt": "4 + 3 =", "answer": "7", "subject": "4 + 3"},
    {"prompt": "1 + 1 =", "answer": "2", "subject": "1 + 1"},
    # Subtraction (10 examples)
    {"prompt": "9 - 4 =", "answer": "5", "subject": "9 - 4"},
    {"prompt": "15 - 7 =", "answer": "8", "subject": "15 - 7"},
    {"prompt": "12 - 5 =", "answer": "7", "subject": "12 - 5"},
    {"prompt": "8 - 3 =", "answer": "5", "subject": "8 - 3"},
    {"prompt": "10 - 6 =", "answer": "4", "subject": "10 - 6"},
    {"prompt": "7 - 2 =", "answer": "5", "subject": "7 - 2"},
    {"prompt": "14 - 9 =", "answer": "5", "subject": "14 - 9"},
    {"prompt": "11 - 8 =", "answer": "3", "subject": "11 - 8"},
    {"prompt": "6 - 1 =", "answer": "5", "subject": "6 - 1"},
    {"prompt": "13 - 6 =", "answer": "7", "subject": "13 - 6"},
    # Multiplication (15 examples)
    {"prompt": "3 \u00d7 4 =", "answer": "12", "subject": "3 \u00d7 4"},
    {"prompt": "7 \u00d7 6 =", "answer": "42", "subject": "7 \u00d7 6"},
    {"prompt": "9 \u00d7 7 =", "answer": "63", "subject": "9 \u00d7 7"},
    {"prompt": "8 \u00d7 6 =", "answer": "48", "subject": "8 \u00d7 6"},
    {"prompt": "5 \u00d7 5 =", "answer": "25", "subject": "5 \u00d7 5"},
    {"prompt": "2 \u00d7 8 =", "answer": "16", "subject": "2 \u00d7 8"},
    {"prompt": "4 \u00d7 7 =", "answer": "28", "subject": "4 \u00d7 7"},
    {"prompt": "6 \u00d7 3 =", "answer": "18", "subject": "6 \u00d7 3"},
    {"prompt": "9 \u00d7 9 =", "answer": "81", "subject": "9 \u00d7 9"},
    {"prompt": "8 \u00d7 7 =", "answer": "56", "subject": "8 \u00d7 7"},
    {"prompt": "3 \u00d7 9 =", "answer": "27", "subject": "3 \u00d7 9"},
    {"prompt": "5 \u00d7 6 =", "answer": "30", "subject": "5 \u00d7 6"},
    {"prompt": "4 \u00d7 4 =", "answer": "16", "subject": "4 \u00d7 4"},
    {"prompt": "7 \u00d7 8 =", "answer": "56", "subject": "7 \u00d7 8"},
    {"prompt": "2 \u00d7 7 =", "answer": "14", "subject": "2 \u00d7 7"},
    # Division (10 examples)
    {"prompt": "12 / 4 =", "answer": "3", "subject": "12 / 4"},
    {"prompt": "45 / 9 =", "answer": "5", "subject": "45 / 9"},
    {"prompt": "24 / 6 =", "answer": "4", "subject": "24 / 6"},
    {"prompt": "36 / 9 =", "answer": "4", "subject": "36 / 9"},
    {"prompt": "18 / 3 =", "answer": "6", "subject": "18 / 3"},
    {"prompt": "21 / 7 =", "answer": "3", "subject": "21 / 7"},
    {"prompt": "40 / 8 =", "answer": "5", "subject": "40 / 8"},
    {"prompt": "15 / 5 =", "answer": "3", "subject": "15 / 5"},
    {"prompt": "56 / 8 =", "answer": "7", "subject": "56 / 8"},
    {"prompt": "27 / 9 =", "answer": "3", "subject": "27 / 9"},
]

HISTORY_DATES = [
    # Wars and conflicts (10)
    {
        "prompt": "World War 2 ended in the year",
        "answer": "1945",
        "subject": "World War 2",
    },
    {
        "prompt": "World War 1 began in the year",
        "answer": "1914",
        "subject": "World War 1",
    },
    {
        "prompt": "The Korean War began in the year",
        "answer": "1950",
        "subject": "Korean War",
    },
    {
        "prompt": "The American Civil War ended in the year",
        "answer": "1865",
        "subject": "American Civil War",
    },
    {
        "prompt": "The Battle of Waterloo took place in the year",
        "answer": "1815",
        "subject": "Battle of Waterloo",
    },
    {
        "prompt": "The Vietnam War ended in the year",
        "answer": "1975",
        "subject": "Vietnam War",
    },
    {
        "prompt": "The Falklands War took place in the year",
        "answer": "1982",
        "subject": "Falklands War",
    },
    {
        "prompt": "The Gulf War began in the year",
        "answer": "1990",
        "subject": "Gulf War",
    },
    {
        "prompt": "The Hundred Years War ended in the year",
        "answer": "1453",
        "subject": "Hundred Years War",
    },
    {
        "prompt": "The Russo-Japanese War began in the year",
        "answer": "1904",
        "subject": "Russo-Japanese War",
    },
    # Political events (10)
    {
        "prompt": "The Declaration of Independence was signed in",
        "answer": "1776",
        "subject": "Declaration of Independence",
    },
    {"prompt": "The Berlin Wall fell in", "answer": "1989", "subject": "Berlin Wall"},
    {
        "prompt": "The French Revolution began in",
        "answer": "1789",
        "subject": "French Revolution",
    },
    {
        "prompt": "The Russian Revolution took place in",
        "answer": "1917",
        "subject": "Russian Revolution",
    },
    {
        "prompt": "The Magna Carta was signed in",
        "answer": "1215",
        "subject": "Magna Carta",
    },
    {
        "prompt": "Nelson Mandela was released from prison in",
        "answer": "1990",
        "subject": "Nelson Mandela",
    },
    {
        "prompt": "The United Nations was founded in",
        "answer": "1945",
        "subject": "United Nations",
    },
    {
        "prompt": "The European Union was established in",
        "answer": "1993",
        "subject": "European Union",
    },
    {
        "prompt": "India gained independence in the year",
        "answer": "1947",
        "subject": "India",
    },
    {
        "prompt": "The reunification of Germany occurred in",
        "answer": "1990",
        "subject": "Germany",
    },
    # Scientific milestones (10)
    {
        "prompt": "The first moon landing occurred in the year",
        "answer": "1969",
        "subject": "moon landing",
    },
    {
        "prompt": "Penicillin was discovered in the year",
        "answer": "1928",
        "subject": "Penicillin",
    },
    {
        "prompt": "The first nuclear bomb was detonated in the year",
        "answer": "1945",
        "subject": "nuclear bomb",
    },
    {
        "prompt": "The structure of DNA was discovered in the year",
        "answer": "1953",
        "subject": "DNA",
    },
    {
        "prompt": "The first heart transplant was performed in",
        "answer": "1967",
        "subject": "heart transplant",
    },
    {
        "prompt": "The World Wide Web was invented in the year",
        "answer": "1989",
        "subject": "World Wide Web",
    },
    {
        "prompt": "The Wright brothers first flew in",
        "answer": "1903",
        "subject": "Wright brothers",
    },
    {
        "prompt": "Einstein published general relativity in",
        "answer": "1915",
        "subject": "Einstein",
    },
    {
        "prompt": "The Hubble Space Telescope was launched in the year",
        "answer": "1990",
        "subject": "Hubble Space Telescope",
    },
    {
        "prompt": "Dolly the sheep was cloned in",
        "answer": "1996",
        "subject": "Dolly the sheep",
    },
    # Cultural/social milestones (10)
    {"prompt": "The Titanic sank in the year", "answer": "1912", "subject": "Titanic"},
    {
        "prompt": "Women gained the right to vote in the US in",
        "answer": "1920",
        "subject": "Women",
    },
    {
        "prompt": "The first modern Olympic Games were in",
        "answer": "1896",
        "subject": "Olympic Games",
    },
    {
        "prompt": "MLK gave the I Have a Dream speech in",
        "answer": "1963",
        "subject": "MLK",
    },
    {
        "prompt": "Slavery was abolished in the US in",
        "answer": "1865",
        "subject": "Slavery",
    },
    {
        "prompt": "The first iPhone was released in the year",
        "answer": "2007",
        "subject": "iPhone",
    },
    {
        "prompt": "Gutenberg invented the printing press around",
        "answer": "1440",
        "subject": "Gutenberg",
    },
    {
        "prompt": "Shakespeare was born in the year",
        "answer": "1564",
        "subject": "Shakespeare",
    },
    {
        "prompt": "The Great Fire of London was in",
        "answer": "1666",
        "subject": "Great Fire of London",
    },
    {
        "prompt": "The Panama Canal was completed in the year",
        "answer": "1914",
        "subject": "Panama Canal",
    },
]

SCIENCE_SYMBOLS = [
    # Chemical symbols (20)
    {"prompt": "The chemical symbol for gold is", "answer": "Au", "subject": "gold"},
    {"prompt": "The chemical symbol for iron is", "answer": "Fe", "subject": "iron"},
    {
        "prompt": "The chemical symbol for silver is",
        "answer": "Ag",
        "subject": "silver",
    },
    {
        "prompt": "The chemical symbol for copper is",
        "answer": "Cu",
        "subject": "copper",
    },
    {
        "prompt": "The chemical symbol for sodium is",
        "answer": "Na",
        "subject": "sodium",
    },
    {
        "prompt": "The chemical symbol for potassium is",
        "answer": "K",
        "subject": "potassium",
    },
    {
        "prompt": "The chemical symbol for mercury is",
        "answer": "Hg",
        "subject": "mercury",
    },
    {"prompt": "The chemical symbol for lead is", "answer": "Pb", "subject": "lead"},
    {"prompt": "The chemical symbol for tin is", "answer": "Sn", "subject": "tin"},
    {
        "prompt": "The chemical symbol for tungsten is",
        "answer": "W",
        "subject": "tungsten",
    },
    {
        "prompt": "The chemical symbol for helium is",
        "answer": "He",
        "subject": "helium",
    },
    {"prompt": "The chemical symbol for neon is", "answer": "Ne", "subject": "neon"},
    {"prompt": "The chemical symbol for argon is", "answer": "Ar", "subject": "argon"},
    {"prompt": "The chemical symbol for oxygen is", "answer": "O", "subject": "oxygen"},
    {
        "prompt": "The chemical symbol for nitrogen is",
        "answer": "N",
        "subject": "nitrogen",
    },
    {"prompt": "The chemical symbol for carbon is", "answer": "C", "subject": "carbon"},
    {
        "prompt": "The chemical symbol for hydrogen is",
        "answer": "H",
        "subject": "hydrogen",
    },
    {
        "prompt": "The chemical symbol for calcium is",
        "answer": "Ca",
        "subject": "calcium",
    },
    {"prompt": "The chemical symbol for zinc is", "answer": "Zn", "subject": "zinc"},
    {
        "prompt": "The chemical symbol for platinum is",
        "answer": "Pt",
        "subject": "platinum",
    },
    # Physical constants (10)
    {
        "prompt": "The speed of light in m/s is approximately",
        "answer": "299",
        "subject": "speed of light",
    },
    {
        "prompt": "Absolute zero in Celsius is",
        "answer": "-273",
        "subject": "Absolute zero",
    },
    {
        "prompt": "Acceleration due to gravity in m/s2 is about",
        "answer": "9",
        "subject": "gravity",
    },
    {
        "prompt": "The number of planets in our solar system is",
        "answer": "8",
        "subject": "planets",
    },
    {
        "prompt": "The charge of an electron in units of e is",
        "answer": "-1",
        "subject": "electron",
    },
    {"prompt": "The atomic number of carbon is", "answer": "6", "subject": "carbon"},
    {"prompt": "The atomic number of oxygen is", "answer": "8", "subject": "oxygen"},
    {
        "prompt": "The atomic number of hydrogen is",
        "answer": "1",
        "subject": "hydrogen",
    },
    {"prompt": "The atomic number of gold is", "answer": "79", "subject": "gold"},
    {"prompt": "The atomic number of iron is", "answer": "26", "subject": "iron"},
    # Basic science facts (10)
    {"prompt": "Water boils at", "answer": "100", "subject": "Water"},
    {"prompt": "Water freezes at", "answer": "0", "subject": "Water"},
    {
        "prompt": "The number of chromosomes in a human cell is",
        "answer": "46",
        "subject": "chromosomes",
    },
    {"prompt": "The pH of pure water is", "answer": "7", "subject": "water"},
    {
        "prompt": "The speed of sound in m/s at sea level is about",
        "answer": "343",
        "subject": "speed of sound",
    },
    {
        "prompt": "The number of bones in an adult human body is",
        "answer": "206",
        "subject": "bones",
    },
    {
        "prompt": "The boiling point of nitrogen in Celsius is",
        "answer": "-196",
        "subject": "nitrogen",
    },
    {
        "prompt": "The melting point of iron in Celsius is about",
        "answer": "1538",
        "subject": "iron",
    },
    {
        "prompt": "The density of water in kg/m3 is",
        "answer": "1000",
        "subject": "water",
    },
    {
        "prompt": "Normal human body temperature in Fahrenheit is",
        "answer": "98",
        "subject": "body temperature",
    },
]

RELATION_KEYWORDS: dict[str, set[str]] = {
    "capitals": {"capital"},
    "science_symbols": {"symbol", "number", "atomic", "chemical"},
    "math_simple": set(),
    "history_dates": set(),
}

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


def format_prompt(raw_prompt: str, tokenizer) -> str:
    """Wrap a raw completion prompt in the model's chat template."""
    messages = [{"role": "user", "content": raw_prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


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
