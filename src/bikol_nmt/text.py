import re
import unicodedata


def clean_text(text):
    text = unicodedata.normalize("NFC", text).replace("’", "'").replace("‘", "'")
    return " ".join(text.split())


def fold(text):
    return clean_text(text).lower()


def lookup_key(text):
    return fold(text).strip(" .,!?:;\"“”()[]{}").strip()


def normalize(text):
    return " ".join(re.sub(r"([.,!?;:])", r" \1 ", fold(text)).split())


def display(text):
    text = re.sub(r"\s+([.,!?;:])", r"\1", " ".join(text.split()))
    text = re.sub(r"([([{])\s+", r"\1", text)
    return text[:1].upper() + text[1:]
