""" from https://github.com/keithito/tacotron """
from matcha.text import cleaners
from matcha.text.symbols import symbols

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}  # pylint: disable=unnecessary-comprehension


class UnknownCleanerException(Exception):
    pass


def text_to_sequence(text, cleaner_names, language_code=None):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []

    if language_code:
        clean_text = _clean_text_ml(text, cleaner_names[0], language_code)
        for symbol in clean_text:
            symbol_id = _symbol_to_id[symbol]
            sequence += [symbol_id]
        return sequence, clean_text
    else:
        clean_text = _clean_text(text, cleaner_names)
        for symbol in clean_text:
            symbol_id = _symbol_to_id[symbol]
            sequence += [symbol_id]
        return sequence, clean_text


def cleaned_text_to_sequence(cleaned_text):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
    return sequence


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        s = _id_to_symbol[symbol_id]
        result += s
    return result


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise UnknownCleanerException(f"Unknown cleaner: {name}")
        text = cleaner(text)
    return text


def _clean_text_ml(text, cleaner, language_code):
    cleaner = getattr(cleaners, cleaner)
    text = cleaner(text, language_code)
    return text