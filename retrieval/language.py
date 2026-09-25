"""
Language detection. 
dict of available output languages
Detection is local thru langdetect
"""
from langdetect import detect, LangDetectException

LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "ru": "Russian",
    "hi": "Hindi",
    "mr": "Marathi",
    "ar": "Arabic",
    "bn": "Bengali",
    "zh-cn": "Chinese",
    "ja": "Japanese",
    "pt": "Portuguese",
    "it": "Italian",
    "ko": "Korean",
    "ta": "Tamil",
    "te": "Telugu",
    "gu": "Gujarati",
    "ur": "Urdu",
}

AUTO_DETECT = "auto"
_MIN_CHARS_FOR_DETECTION = 20
_MIN_CONFIDENCE = 0.7

def detect_language(text: str) -> str:
    """Return a language code for `text`, defaulting to English on failure."""
    # try:
    #     code = detect(text)
    # except LangDetectException:
    #     return "en"
    # if code.startswith("zh"):
    #     return "zh-cn"
    # return code if code in LANGUAGE_NAMES else code

    text = text.strip()
    if len(text) < _MIN_CHARS_FOR_DETECTION:
        return "en"
    try:
        candidates = detect_langs(text)
    except LangDetectException:
        return "en"
    if not candidates or candidates[0].prob < _MIN_CONFIDENCE:
        return "en"
    code = candidates[0].lang
    if code.startswith("zh"):
        code = "zh-cn"
    return code if code in LANGUAGE_NAMES else "en"


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def dropdown_options() -> list[str]:
    """Options for the Streamlit selectbox: 'Auto-detect' first, then names."""
    return ["Auto-detect (same as question)"] + [
        f"{name} ({code})" for code, name in LANGUAGE_NAMES.items()
    ]


def code_from_dropdown(choice: str) -> str | None:
    """Map a dropdown label back to a language code, or None for auto-detect."""
    if choice.startswith("Auto-detect"):
        return None
    code = choice.split("(")[-1].rstrip(")")
    return code
