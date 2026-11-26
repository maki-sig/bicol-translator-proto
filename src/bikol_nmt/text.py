import re
import unicodedata


def fold(text):
    text = unicodedata.normalize("NFC", text).replace("’", "'").replace("‘", "'")
    return " ".join(text.lower().split())


def lookup_key(text):
    return fold(text).strip(" .,!?:;\"“”()[]{}").strip()


def normalize(text):
    return " ".join(re.sub(r"([.,!?;:])", r" \1 ", fold(text)).split())


def display(text):
    text = re.sub(r"\s+([.,!?;:])", r"\1", " ".join(text.split()))
    text = re.sub(r"([([{])\s+", r"\1", text)
    return text[:1].upper() + text[1:]
