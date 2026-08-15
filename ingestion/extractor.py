import fitz

VERSION = "pymupdf-v1"


def extract_text_from_pdf(file_bytes):
    """Opens a PDF from raw bytes and extracts all text, page by page."""
    with fitz.open(stream=file_bytes,filetype="pdf") as doc:
        full_text=""
        for page in doc:
            full_text+= page.get_text()
    return full_text


def measure_extraction_quality(text: str) -> dict:
    """
    Measures signals of extraction quality from the flat text string. None
    of these prove correctness on their own - they're heuristics, not a
    pass/fail verdict - but each flags a specific, plausible failure mode
    that would otherwise pass through silently:

    - garbage_char_ratio: fraction of characters that are control
      characters or the Unicode replacement character. A high ratio
      suggests encoding failure or a corrupted/garbled extraction.
    - newline_density: newlines per 1000 characters. Near-zero on a
      multi-page document suggests extraction flattened all structure
      (paragraphs, line breaks) into one undifferentiated blob - the
      "just flattening it to a string" failure mode.
    - avg_word_length: mean length of whitespace-split "words". Extraction
      that loses word boundaries (missing spaces, broken encoding) tends
      to produce abnormally long or short average word lengths compared
      to normal English prose (roughly 4-6 characters).
    """
    if not text:
        return {
            "garbage_char_ratio": 0.0,
            "newline_density": 0.0,
            "avg_word_length": 0.0,
        }

    total_chars = len(text)

    garbage_count = sum(
        1 for ch in text
        if ch == "\ufffd" or (ord(ch) < 32 and ch not in ("\n", "\t", "\r"))
    )
    garbage_char_ratio = round(garbage_count / total_chars, 4)

    newline_count = text.count("\n")
    newline_density = round((newline_count / total_chars) * 1000, 2)

    words = text.split()
    avg_word_length = round(sum(len(w) for w in words) / len(words), 2) if words else 0.0

    return {
        "garbage_char_ratio": garbage_char_ratio,
        "newline_density": newline_density,
        "avg_word_length": avg_word_length,
    }