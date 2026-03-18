from __future__ import annotations

import re
from typing import Any, Dict


HOOK_TRIGGERS = {
    # Price / money tension
    "مين": 12,
    "بكام": 10,
    "جنيه": 12,
    "بس": 4,
    "استنى": 4,
    "في منطقتك": 10,
    "تيك توك": 5,
    "شيف": 5,
    "بيت": 5,
    "رمضان": 8,
    "افطار": 6,
    "سحور": 6,
    # Karim-series character hooks
    "كريم": 8,
    "فارس": 4,
    "أم كريم": 6,
    "الشيف حاتم": 5,
    "نور": 3,
    "سلمى": 3,
    # Story-arc tension words
    "فشل": 6,
    "اترفد": 8,
    "رفضوه": 7,
    "الورشة": 5,
    "المطعم": 5,
    "2otbo5ly": 10,
    "الطلب": 5,
    "البداية": 4,
    "مشاهد": 3,
    # Egyptian life authenticity signals
    "شبرا": 5,
    "المطبخ": 4,
    "السطح": 3,
    "السوشيال": 3,
    "لايف": 3,
}

AD_LANGUAGE_PENALTIES = {
    "يمكنك": -12,
    "من خلال": -10,
    "منصتنا": -12,
    "حل مبتكر": -10,
    "انضم": -8,
    "اكتشف": -8,
    "تجربة فريدة": -10,
    "الآن": -6,
    # Wrong character names from previous pipeline
    "زياد": -15,   # Old name — correct name is فارس
    "خالد": -8,
    "أحمد": -5,
}


def _normalize_arabic(text: str) -> str:
    normalized = text.lower()
    normalized = normalized.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    normalized = normalized.replace("ة", "ه").replace("ى", "ي")
    return normalized


def _count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def score_script_heuristics(script: Dict[str, Any]) -> Dict[str, Any]:
    hook = str(script.get("hook", ""))
    voiceover = str(script.get("voiceover", ""))
    caption = str(script.get("primary_caption", ""))
    cliffhanger = str(script.get("cliffhanger", ""))
    continuity_updates = script.get("continuity_updates", [])
    text_blob = " ".join([hook, voiceover, caption, cliffhanger])
    normalized = _normalize_arabic(text_blob)

    score = 35
    reasons = []

    for token, value in HOOK_TRIGGERS.items():
        if token in normalized:
            score += value
            reasons.append(f"+{value} token:{token}")

    for token, value in AD_LANGUAGE_PENALTIES.items():
        if token in normalized:
            score += value
            reasons.append(f"{value} token:{token}")

    hook_words = _count_words(hook)
    if 4 <= hook_words <= 10:
        score += 10
        reasons.append("+10 hook_length")
    else:
        score -= 6
        reasons.append("-6 hook_length")

    voice_words = _count_words(voiceover)
    if 30 <= voice_words <= 75:
        score += 8
        reasons.append("+8 voiceover_length")
    else:
        score -= 6
        reasons.append("-6 voiceover_length")

    if "?" in hook or "؟" in hook:
        score += 6
        reasons.append("+6 question_hook")

    if any(char.isdigit() for char in hook):
        score += 6
        reasons.append("+6 numeric_hook")

    if cliffhanger:
        score += 5
        reasons.append("+5 cliffhanger_present")
        if "?" in cliffhanger or "؟" in cliffhanger:
            score += 4
            reasons.append("+4 cliffhanger_question")

    if continuity_updates:
        score += 4
        reasons.append("+4 continuity_updates")

    # Reward correct episode number being set in script
    episode_number = script.get("episode_number", 0)
    if episode_number and int(episode_number) > 0:
        score += 5
        reasons.append(f"+5 episode_number_set:{episode_number}")

    # Reward episode fidelity when the script explicitly names the correct protagonist
    if "كريم" in normalized:
        score += 4
        reasons.append("+4 protagonist_named")

    # Penalize wrong character names hard — these break visual consistency
    if "زياد" in normalized:
        score -= 20
        reasons.append("-20 wrong_character_ziad")

    score = max(0, min(100, score))
    return {"total": score, "reasons": reasons}

