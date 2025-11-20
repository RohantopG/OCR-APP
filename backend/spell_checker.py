# spell_checker.py
from symspellpy import SymSpell, Verbosity
import os

def load_symspell():
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

    # Use absolute path relative to this script's directory
    base_path = os.path.dirname(__file__)
    dictionary_path = os.path.join(base_path, "kannada_wordList_with_freq.txt")

    if not os.path.exists(dictionary_path):
        raise FileNotFoundError(f"Dictionary file not found at: {dictionary_path}")

    # Use load_dictionary instead of load_dictionary_stream
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, encoding='utf-8')

    return sym_spell

def correct_spelling(text, sym_spell):
    words = text.split()
    corrected_text_parts = []

    # Use lookup_compound to handle the whole text at once, it's more efficient
    # and provides better context for corrections.
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)

    # We only accept a suggestion if it's a very close match (distance 1)
    # or if it's a perfect match (distance 0). This prevents aggressive,
    # incorrect changes on noisy OCR output.
    if suggestions and suggestions[0].distance <= 1:
        return suggestions[0].term
    else:
        # If no high-confidence suggestion is found, return the original text.
        return text
