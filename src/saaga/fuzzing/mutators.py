"""Text mutation operators for SAAGA adversarial fuzzing.
=====================================================
Ported and modernized from JailGuard reimplementation.
Provides 10 structure-preserving and perturbation mutators, including:
  - Synonym Replacement (SR) via NLTK WordNet with offline dictionary fallback
  - Punctuation Insertion (PI) via AEDA with manual random insertion fallback
  - Cross-lingual Translation (TL) via NLLB-200 (CPU/CUDA) with graceful no-op fallback
  - Multi-layer Cipher Encoding (EN): stacked reversible ciphers (base64, rot13, caesar, leet)
  - Policy combinator (PL)
  - Character perturbation mutators: RR, RI, TR, TI, RD
"""

from __future__ import annotations

import base64
import codecs
import logging
import os
import random
import re
from typing import Callable, Final, Optional, Sequence

logger = logging.getLogger(__name__)

# ─── NLTK Imports & Offline Safety ───────────────────────────────────────────

try:
    import nltk
    from nltk.corpus import stopwords, wordnet
    from nltk.tokenize import sent_tokenize, word_tokenize

    _NLTK_OK = True
except ImportError:
    nltk = None
    stopwords = None
    wordnet = None
    sent_tokenize = None
    word_tokenize = None
    _NLTK_OK = False


def ensure_nltk() -> None:
    """Check for pre-downloaded NLTK resources without triggering network downloads."""
    if not _NLTK_OK or nltk is None:
        return
    packages = [
        ("tokenizers", "punkt_tab"),
        ("tokenizers", "punkt"),
        ("corpora", "stopwords"),
        ("corpora", "wordnet"),
    ]
    for resource_type, pkg in packages:
        try:
            nltk.data.find(f"{resource_type}/{pkg}")
        except (LookupError, OSError):
            pass


try:
    ensure_nltk()
except Exception:
    pass

# ─── PyTorch / CUDA Detection (for NLLB Translation) ─────────────────────────

try:
    import torch

    _TORCH_OK = True
except ImportError:
    torch = None
    _TORCH_OK = False


def _nllb_device() -> str:
    """Resolve the device for NLLB translation inference.

    Honors SAAGA_TL_DEVICE or AUTORED_TL_DEVICE to force CPU or GPU.
    Defaults to cuda:0 when torch sees a GPU, else cpu.
    """
    if not _TORCH_OK or torch is None:
        return "cpu"
    forced = os.environ.get("SAAGA_TL_DEVICE") or os.environ.get("AUTORED_TL_DEVICE", "")
    forced = forced.strip().lower()
    if forced in ("cpu", "0", "cuda:0", "gpu"):
        if forced in ("0", "gpu", "cuda:0"):
            return "cuda:0"
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


class _NullCtx:
    """No-op context manager mirroring torch.no_grad for CPU/torch-absent paths."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return False


# ─── Offline Dictionaries & Stopwords ────────────────────────────────────────

_DEFAULT_STOPWORDS: frozenset[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
})

_OFFLINE_SYNONYMS: Final[dict[str, list[str]]] = {
    "reveal": ["disclose", "show", "unveil", "expose", "display"],
    "secret": ["confidential", "hidden", "private", "classified"],
    "password": ["passcode", "credential", "key", "token"],
    "access": ["entry", "admission", "permission", "clearance"],
    "instruction": ["directive", "command", "guideline", "rule"],
    "instructions": ["directives", "commands", "guidelines", "rules"],
    "system": ["framework", "platform", "structure"],
    "ignore": ["disregard", "bypass", "overlook", "skip"],
    "previous": ["prior", "former", "earlier", "preceding"],
    "rule": ["policy", "guideline", "constraint", "regulation"],
    "rules": ["policies", "guidelines", "constraints", "regulations"],
    "prompt": ["query", "input", "message", "request"],
    "code": ["cipher", "key", "token", "string"],
    "grant": ["allow", "permit", "authorize", "admit"],
    "admin": ["administrator", "supervisor", "root", "operator"],
    "user": ["operator", "client", "person", "account"],
    "bypass": ["circumvent", "sidestep", "avoid", "evade"],
    "explain": ["clarify", "describe", "detail", "explicate"],
    "output": ["print", "produce", "generate", "return"],
    "help": ["assist", "aid", "support"],
    "tell": ["inform", "notify", "state", "relay"],
    "repeat": ["echo", "restate", "iterate", "duplicate"],
    "execute": ["perform", "run", "enact", "carry out"],
    "verify": ["confirm", "validate", "authenticate", "check"],
}

# ─── Language Mapping for Translation ───────────────────────────────────────

_LANG_POOL: Final[list[str]] = ["ru", "fr", "de", "el", "id", "it", "ja", "ko", "pl", "zh"]

_LANG_TO_FLORES: Final[dict[str, str]] = {
    "ru": "rus_Cyrl",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "id": "ind_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "pl": "pol_Latn",
    "zh": "zho_Hans",
    "en": "eng_Latn",
}

_PUNCTS: Final[list[str]] = [".", ",", ";", ":", "!", "?"]


# ─── Helper Functions ────────────────────────────────────────────────────────

def _remove_non_utf8(text: str) -> str:
    """Filter to ASCII-safe code points."""
    return "".join(c for c in text if ord(c) < 128)


def _get_synonyms(word: str) -> list[str]:
    """Retrieve synonyms for a word via WordNet, with offline dictionary fallback."""
    synonyms: set[str] = set()
    cleaned = word.lower().strip()

    if _NLTK_OK and wordnet is not None:
        try:
            for syn in wordnet.synsets(cleaned):
                for lemma in syn.lemmas():
                    s = lemma.name().replace("_", " ").replace("-", " ").lower()
                    s = "".join(c for c in s if c in " abcdefghijklmnopqrstuvwxyz")
                    if s and s != cleaned:
                        synonyms.add(s)
        except Exception:
            pass

    # Offline dictionary fallback
    if not synonyms and cleaned in _OFFLINE_SYNONYMS:
        synonyms.update(_OFFLINE_SYNONYMS[cleaned])

    synonyms.discard(cleaned)
    return list(synonyms)


def _find_important_sentences(text: str, n: int = 3) -> list[tuple[int, int]]:
    """Return character spans (start, end) of the top-n most important sentences."""
    if not text or not text.strip():
        return []

    # Try NLTK sentence tokenizer first
    if _NLTK_OK and sent_tokenize is not None and word_tokenize is not None:
        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= n:
                return []
            try:
                stop = set(stopwords.words("english")) if stopwords is not None else _DEFAULT_STOPWORDS
            except Exception:
                stop = _DEFAULT_STOPWORDS
            freq: dict[str, int] = {}
            for s in sentences:
                for w in word_tokenize(s.lower()):
                    if w.isalnum() and w not in stop:
                        freq[w] = freq.get(w, 0) + 1
            scored = [
                (
                    sum(
                        freq.get(w, 0)
                        for w in word_tokenize(s.lower())
                        if w.isalnum() and w not in stop
                    ),
                    s,
                )
                for s in sentences
            ]
            scored.sort(reverse=True)
            top = [s for _, s in scored[:n]]
            positions = []
            for s in top:
                idx = text.find(s)
                if idx != -1:
                    positions.append((idx, idx + len(s)))
            return positions
        except Exception:
            pass

    # Pure regex fallback for offline environments without NLTK corpora
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) <= n:
        return []

    words_per_s = [s.split() for s in sentences]
    word_freq: dict[str, int] = {}
    for ws in words_per_s:
        for w in ws:
            wl = w.lower().strip(".,!?:;\"'()[]{}")
            if wl and wl not in _DEFAULT_STOPWORDS:
                word_freq[wl] = word_freq.get(wl, 0) + 1

    scored_fallback = []
    for s, ws in zip(sentences, words_per_s):
        score = sum(word_freq.get(w.lower().strip(".,!?:;\"'()[]{}"), 0) for w in ws)
        scored_fallback.append((score, s))
    scored_fallback.sort(reverse=True)
    top = [s for _, s in scored_fallback[:n]]
    positions = []
    for s in top:
        idx = text.find(s)
        if idx != -1:
            positions.append((idx, idx + len(s)))
    return positions


# ═════════════════════════════════════════════════════════════════════════════
#  MUTATOR 1: Random Replacement (RR)
# ═════════════════════════════════════════════════════════════════════════════

def random_replacement(text: str, rate: float = 0.005) -> str:
    """Replace ~rate% of character positions with [Mask] (6-char chunks)."""
    if not text:
        return text
    chars = list(text)
    skip = 0
    i = 0
    while i < len(chars):
        if skip > 0:
            skip -= 1
            i += 1
            continue
        if random.random() < rate:
            chars[i : i + 6] = list("[Mask]")
            skip = 5
        i += 1
    return "".join(chars)


# ═════════════════════════════════════════════════════════════════════════════
#  MUTATOR 2: Random Insertion (RI)
# ═════════════════════════════════════════════════════════════════════════════

def random_insertion(text: str, rate: float = 0.005) -> str:
    """Insert [Mask] tokens at ~rate% of positions (content preserved)."""
    if not text:
        return text
    insert_positions = sorted(
        [i for i in range(len(text)) if random.random() < rate],
        reverse=True,
    )
    result = list(text)
    for pos in insert_positions:
        result.insert(pos, "[Mask]")
    return "".join(result)


# ═════════════════════════════════════════════════════════════════════════════
#  MUTATOR 3: Targeted Replacement (TR)
# ═════════════════════════════════════════════════════════════════════════════

def targeted_replacement(text: str, rate: float = 0.005, boost: int = 5) -> str:
    """Like random_replacement but with boost-higher rate in important sentences."""
    if not text:
        return text
    important = _find_important_sentences(text)
    chars = list(text)
    skip = 0
    i = 0
    while i < len(chars):
        if skip > 0:
            skip -= 1
            i += 1
            continue
        effective_rate = rate
        for s, e in important:
            if s <= i < e:
                effective_rate = rate * boost
                break
        if random.random() < effective_rate:
            chars[i : i + 6] = list("[Mask]")
            skip = 5
        i += 1
    return "".join(chars)


# ═════════════════════════════════════════════════════════════════════════════
#  MUTATOR 4: Targeted Insertion (TI)
# ═════════════════════════════════════════════════════════════════════════════

def targeted_insertion(text: str, rate: float = 0.005, boost: int = 5) -> str:
    """Like random_insertion but with boost-higher rate in important sentences."""
    if not text:
        return text
    important = _find_important_sentences(text)
    insert_positions = []
    for i in range(len(text)):
        effective_rate = rate
        for s, e in important:
            if s <= i < e:
                effective_rate = rate * boost
                break
        if random.random() < effective_rate:
            insert_positions.append(i)
    insert_positions.sort(reverse=True)
    result = list(text)
    for pos in insert_positions:
        result.insert(pos, "[Mask]")
    return "".join(result)


# ═════════════════════════════════════════════════════════════════════════════
#  MUTATOR 5: Random Deletion (RD)
# ═════════════════════════════════════════════════════════════════════════════

def random_deletion(text: str, rate: float = 0.005) -> str:
    """Delete characters with probability `rate` (skip next 5 after each deletion)."""
    if not text:
        return text
    result = []
    skip = 0
    for c in text:
        if skip > 0:
            skip -= 1
            continue
        if random.random() < rate:
            skip = 5
        else:
            result.append(c)
    return "".join(result)


# ═════════════════════════════════════════════════════════════════════════════
#  MUTATOR 6: Synonym Replacement (SR) via NLTK WordNet with offline fallback
# ═════════════════════════════════════════════════════════════════════════════

def synonym_replacement(text: str, n: Optional[int] = None) -> str:
    """Replace up to n random non-stopwords with WordNet or offline synonyms."""
    if not text or not text.strip():
        return text

    try:
        stop = set(stopwords.words("english")) if _NLTK_OK and stopwords is not None else _DEFAULT_STOPWORDS
    except Exception:
        stop = _DEFAULT_STOPWORDS

    words = text.split()
    if not words:
        return text

    if n is None:
        n = max(1, min(20, len(words) // 3))

    candidates = [w for w in words if w.lower().strip(".,!?:;\"'()[]{}") not in stop]
    if not candidates:
        return text

    random.shuffle(candidates)
    replaced = 0
    result = words[:]
    for word in candidates:
        if replaced >= n:
            break
        cleaned_word = word.lower().strip(".,!?:;\"'()[]{}")
        syns = _get_synonyms(cleaned_word)
        if syns:
            syn = random.choice(syns)
            # Preserve trailing punctuation if present
            trailing = ""
            for ch in reversed(word):
                if ch in ".,!?:;\"'()[]{}":
                    trailing = ch + trailing
                else:
                    break
            new_word = syn + trailing if trailing else syn
            result = [new_word if w == word else w for w in result]
            replaced += 1

    return " ".join(result)


# ═════════════════════════════════════════════════════════════════════════════
#  MUTATOR 7: Punctuation Insertion (PI) via AEDA
# ═════════════════════════════════════════════════════════════════════════════

def punctuation_insertion(text: str) -> str:
    """Insert random punctuation marks between words (AEDA-style)."""
    if not text or not text.strip():
        return text

    try:
        from textaugment import AEDA

        t = AEDA()
        augmented = t.punct_insertion(text)
        if augmented and augmented.strip():
            return augmented
    except Exception:
        pass

    # Pure Python manual fallback
    words = text.split()
    if not words:
        return text
    result = []
    for w in words:
        result.append(w)
        if random.random() < 0.15:
            result.append(random.choice(_PUNCTS))
    return " ".join(result)


# ═════════════════════════════════════════════════════════════════════════════
#  MUTATOR 8: Translation (TL) via NLLB-200 with graceful no-op fallback
# ═════════════════════════════════════════════════════════════════════════════

_TL_WARNED = False
_NLLB_MODEL = None
_NLLB_TOK = None
_NLLB_TRIED = False
_NLLB_DEVICE = "cpu"


def _load_nllb():
    """Lazy-load local NLLB-200 model + tokenizer for offline translation."""
    global _NLLB_MODEL, _NLLB_TOK, _NLLB_TRIED, _NLLB_DEVICE
    if _NLLB_TRIED:
        return _NLLB_MODEL, _NLLB_TOK
    _NLLB_TRIED = True

    try:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        candidates = [
            os.environ.get("SAAGA_NLLB_PATH"),
            "models/translation",
            "models/nllb-200-distilled-600M",
            "facebook/nllb-200-distilled-600M",
        ]
        name = "facebook/nllb-200-distilled-600M"
        for c in candidates:
            if c and (os.path.exists(c) or c == "facebook/nllb-200-distilled-600M"):
                name = c
                break
        _NLLB_TOK = AutoTokenizer.from_pretrained(name, local_files_only=True)
        device = _nllb_device()

        torch_dtype = None
        if device != "cpu" and _TORCH_OK and torch is not None:
            torch_dtype = torch.float16

        _NLLB_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
            name, local_files_only=True, torch_dtype=torch_dtype
        )
        _NLLB_DEVICE = device
        if device != "cpu":
            try:
                _NLLB_MODEL = _NLLB_MODEL.to(device)
            except Exception as de:
                logger.warning("[TL] NLLB GPU placement failed (%s); falling back to CPU.", de)
                _NLLB_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
                    name, local_files_only=True
                )
                _NLLB_DEVICE = "cpu"
        _NLLB_MODEL.eval()
        logger.info("[TL] NLLB-200 loaded (device=%s) for TL mutator.", _NLLB_DEVICE)
    except Exception as e:
        _NLLB_MODEL = None
        _NLLB_TOK = None
        _NLLB_DEVICE = "cpu"
        logger.info("[TL] Local NLLB unavailable (%s); will try online backends or no-op.", e)

    return _NLLB_MODEL, _NLLB_TOK


def translation(text: str, target_lang: Optional[str] = None) -> str:
    """Translate text to a target language — offline via NLLB-200 with graceful fallback."""
    global _TL_WARNED
    if not text or not text.strip():
        return text

    lang = target_lang or random.choice(_LANG_POOL)
    clean_text = _remove_non_utf8(text)
    if not clean_text.strip():
        return text

    # 1. Local NLLB-200 (preferred offline backend)
    model, tok = _load_nllb()
    if model is not None and tok is not None:
        flores = _LANG_TO_FLORES.get(lang)
        if not flores:
            return text
        try:
            enc = tok(clean_text, return_tensors="pt")
            device = _NLLB_DEVICE
            if device != "cpu" and _TORCH_OK and torch is not None:
                enc = {k: v.to(device) for k, v in enc.items()}
            bos = tok.convert_tokens_to_ids(flores)
            no_grad = torch.no_grad() if _TORCH_OK and torch is not None else _NullCtx()
            with no_grad:
                out = model.generate(
                    **enc, forced_bos_token_id=bos, max_length=200, num_beams=2
                )
            if device != "cpu" and _TORCH_OK and hasattr(out, "cpu"):
                out = out.cpu()
            translated = tok.batch_decode(out, skip_special_tokens=True)[0]
            if translated and translated.strip():
                return translated
            return text
        except Exception as e:
            if not _TL_WARNED:
                logger.warning("[TL] NLLB translate failed (%s); trying fallback.", e)
                _TL_WARNED = True

    # 2. textaugment Translate fallback (online)
    try:
        from textaugment import Translate

        t = Translate(src="en", to=lang)
        augmented = t.augment(clean_text)
        if augmented and augmented.strip():
            return augmented
    except Exception:
        pass

    # 3. deep_translator fallback (online)
    try:
        from deep_translator import GoogleTranslator

        translated = GoogleTranslator(source="en", target=lang).translate(clean_text)
        if translated and translated.strip():
            return translated
    except Exception:
        pass

    # 4. Graceful no-op fallback: return original text unchanged
    if not _TL_WARNED:
        logger.info("[TL] No translation backend available; returning original text unchanged.")
        _TL_WARNED = True
    return text


# ═════════════════════════════════════════════════════════════════════════════
#  MUTATOR 9: Policy Combination (PL)
# ═════════════════════════════════════════════════════════════════════════════

_POLICY_DEFAULT: Final[dict[str, Any]] = {
    "pool": ["PI", "TI", "TL"],
    "probs": [0.24, 0.52, 0.24],
}


def policy(
    text: str,
    pool: Optional[Sequence[str]] = None,
    probs: Optional[Sequence[float]] = None,
) -> str:
    """Randomly select one mutator from the pool using the given probabilities."""
    p_pool = list(pool or _POLICY_DEFAULT["pool"])
    p_probs = list(probs or _POLICY_DEFAULT["probs"])
    assert abs(sum(p_probs) - 1.0) < 1e-5, "Probabilities must sum to 1.0"

    r = random.random()
    cumulative = 0.0
    chosen = p_pool[-1]
    for name, p in zip(p_pool, p_probs):
        cumulative += p
        if r < cumulative:
            chosen = name
            break

    return apply_mutator(text, chosen)


# ═════════════════════════════════════════════════════════════════════════════
#  MUTATOR 10: Multi-layer Cipher Encoding (EN)
# ═════════════════════════════════════════════════════════════════════════════

_LEET_MAP: Final[dict[str, str]] = {
    "a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7",
    "A": "4", "E": "3", "I": "1", "O": "0", "S": "5", "T": "7",
}


def _enc_rot13(t: str) -> str:
    return codecs.encode(t, "rot13")


def _dec_rot13(t: str) -> str:
    return codecs.decode(t, "rot13")


def _enc_base64(t: str) -> str:
    return base64.b64encode(t.encode("utf-8")).decode("ascii")


def _dec_base64(t: str) -> str:
    return base64.b64decode(t.encode("ascii")).decode("utf-8")


def _enc_leet(t: str) -> str:
    return "".join(_LEET_MAP.get(c, c) for c in t)


def _dec_leet(t: str) -> str:
    rev = {v: k for k, v in _LEET_MAP.items()}
    return "".join(rev.get(c, c) for c in t)


def _enc_caesar(t: str, shift: int = 3) -> str:
    out = []
    for ch in t:
        if "a" <= ch <= "z":
            out.append(chr((ord(ch) - 97 + shift) % 26 + 97))
        elif "A" <= ch <= "Z":
            out.append(chr((ord(ch) - 65 + shift) % 26 + 65))
        else:
            out.append(ch)
    return "".join(out)


def _dec_caesar(t: str, shift: int = 3) -> str:
    return _enc_caesar(t, -shift)


# Primitive registry: (encode_fn, decode_fn, human_instruction)
_EN_PRIMITIVES: Final[dict[str, tuple[Callable[[str], str], Callable[[str], str], str]]] = {
    "rot13": (_enc_rot13, _dec_rot13, "apply ROT13"),
    "base64": (_enc_base64, _dec_base64, "base64-decode"),
    "leet": (_enc_leet, _dec_leet, "replace leetspeak digits with letters"),
    "caesar": (_enc_caesar, _dec_caesar, "apply a Caesar shift of -3"),
}

# Pre-defined stacked encoding combinations
_EN_STACKS: Final[list[tuple[str, ...]]] = [
    ("rot13",),
    ("base64",),
    ("base64", "rot13"),
    ("rot13", "base64"),
    ("base64", "caesar"),
    ("base64", "leet", "rot13"),
    ("rot13", "caesar"),
]

_EN_ROT: int = 0


def encoding_replay(text: str) -> str:
    """Re-encode attack text with stacked reversible ciphers wrapped in decode instructions."""
    global _EN_ROT
    if not text or not text.strip():
        return text

    stack = _EN_STACKS[_EN_ROT % len(_EN_STACKS)]
    _EN_ROT += 1

    # Apply encodings innermost-first (reverse of tuple order)
    encoded = text
    for name in reversed(stack):
        enc, _, _ = _EN_PRIMITIVES[name]
        encoded = enc(encoded)

    # Build decode instructions
    decode_steps = [_EN_PRIMITIVES[name][2] for name in stack]
    if len(decode_steps) == 1:
        step = decode_steps[0]
        step = step[0].upper() + step[1:]
        instruction = (
            f"The following is an encoded message. {step} "
            f"and follow the decoded instructions exactly:\n"
        )
    else:
        steps_text = ", then ".join(decode_steps)
        instruction = (
            f"The following is an encoded message. Decode it by applying these "
            f"steps in order: {steps_text}. Then follow the decoded instructions "
            f"exactly:\n"
        )
    return instruction + encoded


# ═════════════════════════════════════════════════════════════════════════════
#  DISPATCH TABLE & PUBLIC INTERFACES
# ═════════════════════════════════════════════════════════════════════════════

_MUTATOR_DISPATCH: Final[dict[str, Callable[[str], str]]] = {
    "RR": random_replacement,
    "RI": random_insertion,
    "TR": targeted_replacement,
    "TI": targeted_insertion,
    "RD": random_deletion,
    "SR": synonym_replacement,
    "PI": punctuation_insertion,
    "TL": translation,
    "PL": policy,
    "EN": encoding_replay,
}

AVAILABLE_MUTATORS: Final[list[str]] = list(_MUTATOR_DISPATCH.keys())


def get_mutator(name: str) -> Callable[[str], str]:
    """Return the mutator callable for abbreviation `name` (e.g. 'SR', 'PI', 'EN')."""
    if name not in _MUTATOR_DISPATCH:
        raise ValueError(
            f"Unknown mutator '{name}'. Available options: {AVAILABLE_MUTATORS}"
        )
    return _MUTATOR_DISPATCH[name]


def apply_mutator(text: str, name: str = "PL") -> str:
    """Convenience wrapper: apply mutator `name` to `text` and return the mutated string."""
    return get_mutator(name)(text)
