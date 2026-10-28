import random
import re
import pandas as pd
import numpy as np

LEET_MAP = {
    "a": ["@", "4"], "e": ["3"], "i": ["1", "!"],
    "o": ["0"], "s": ["$", "5"], "t": ["7"]
}

def perturb_leetspeak(text: str, probability: float = 0.4) -> str:
    res = []
    for c in text:
        lower_c = c.lower()
        if lower_c in LEET_MAP and random.random() < probability:
            res.append(random.choice(LEET_MAP[lower_c]))
        else:
            res.append(c)
    return "".join(res)

def perturb_spacing(text: str) -> str:
    words = text.split()
    res = []
    for w in words:
        if len(w) > 3 and random.random() < 0.3:
            res.append(" ".join(list(w)))
        else:
            res.append(w)
    return " ".join(res)

def perturb_typos(text: str) -> str:
    words = text.split()
    res = []
    for w in words:
        if len(w) > 4 and random.random() < 0.35:
            # Swap two adjacent letters or drop one
            idx = random.randint(1, len(w) - 2)
            w_mod = w[:idx] + w[idx+1] + w[idx] + w[idx+2:]
            res.append(w_mod)
        else:
            res.append(w)
    return " ".join(res)

def perturb_obfuscation(text: str) -> str:
    # Inserts dots, dashes, or masks URLs
    text = re.sub(r"\b(free|winner|prize|cash|claim)\b", lambda m: ".".join(list(m.group(1))), text, flags=re.IGNORECASE)
    text = text.replace("http://", "hxxp://").replace("https://", "hxxps://")
    return text

def apply_all_perturbations(text: str) -> dict:
    return {
        "Original": text,
        "Leetspeak Substituted": perturb_leetspeak(text),
        "Spaced Characters": perturb_spacing(text),
        "Typos & Swaps": perturb_typos(text),
        "Evasive Obfuscation": perturb_obfuscation(text)
    }
