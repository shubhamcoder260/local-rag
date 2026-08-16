"""
chunker.py

Responsible for splitting raw text into retrieval-ready chunks.

This module performs no embedding, storage, or document reading.
It simply converts text into overlapping chunks that are later
embedded and indexed.
"""

import re

VERSION = "recursive-semantic-v3"

DEFAULT_MAX_CHARS = 1500
DEFAULT_OVERLAP_CHARS = 150


def chunk_text(text, max_chars=DEFAULT_MAX_CHARS, overlap_chars=DEFAULT_OVERLAP_CHARS):
    """
    Splits text recursively using semantic separators to preserve sentence and
    paragraph structures whenever possible, falling back to word/character
    splitting only when necessary.

    Separator hierarchy (tried in order):
        1. Double newline  — paragraph boundary
        2. Single newline  — line boundary
        3. Period + space  — sentence boundary (lookbehind, keeps the period)
        4. Space           — word boundary
        5. Hard slice      — last resort for unbreakable tokens
    """
    separators = ["\n\n", "\n", r"(?<=\. )", " ", ""]
    return _split_text(text, separators, max_chars, overlap_chars)


def _split_text(text, separators, max_chars, overlap_chars):
    """
    Internal recursive worker. Tries each separator in `separators` in order,
    using the first one that actually appears in `text`. Chunks that still
    exceed `max_chars` after splitting are passed back into this function with
    the next (finer) separator level.

    Overlap is preserved across recursion boundaries: before flushing a
    completed chunk and recursing into an oversized split, the overlap tail of
    the current accumulator is prepended to the oversized split so that the
    first sub-chunk produced by the recursive call has continuity with the
    chunk that was just emitted.
    """
    if len(text) <= max_chars:
        return [text]

    # Walk the separator list to find the coarsest one present in this text.
    separator = separators[-1]
    new_separators = []
    for i, _s in enumerate(separators):
        if _s == "":
            separator = _s
            break
        if re.search(_s, text):
            separator = _s
            new_separators = separators[i + 1:]
            break

    # Hard character-level slice — absolute last resort.
    if separator == "":
        chunks = []
        for i in range(0, len(text), max_chars - overlap_chars):
            chunks.append(text[i:i + max_chars])
        return chunks

    # Split on the chosen semantic boundary.
    if separator == r"(?<=\. )":
        splits = re.split(separator, text)
        separator_str = ""
    else:
        splits = text.split(separator)
        separator_str = separator

    splits = [s for s in splits if s]  # drop empty strings

    chunks = []
    current_chunk = []
    current_len = 0

    for split in splits:
        # ------------------------------------------------------------------
        # Oversized split: this single piece is larger than max_chars and
        # must itself be recursed upon with the next finer separator.
        # ------------------------------------------------------------------
        if len(split) > max_chars:
            if current_chunk:
                chunks.append(separator_str.join(current_chunk))

                # Compute overlap tail — identical logic to the normal break
                # path below. Prepend it to the oversized split BEFORE
                # recursing so the first sub-chunk the recursive call produces
                # has continuity with the chunk we just flushed. Without this
                # the boundary between the flushed chunk and sub_chunks[0]
                # would have near-zero overlap.
                overlap_chunk = []
                overlap_len = 0
                for w in reversed(current_chunk):
                    w_len = len(w) + (len(separator_str) if overlap_chunk else 0)
                    if overlap_len + w_len > overlap_chars:
                        break
                    overlap_chunk.insert(0, w)
                    overlap_len += w_len

                if not overlap_chunk:
                    # Every unit exceeds overlap_chars (e.g. long sentences at
                    # the sentence-split level). Fall back to a char-level suffix
                    # of the last unit so the boundary is not completely cold.
                    tail = current_chunk[-1][-overlap_chars:]
                    overlap_chunk = [tail]
                    overlap_len = len(tail)

                if overlap_chunk:
                    overlap_str = separator_str.join(overlap_chunk)
                    # If separator_str is "" (sentence-regex branch) use a
                    # single space as the joiner to avoid merging words.
                    joiner = separator_str if separator_str else " "
                    split = overlap_str + joiner + split

                current_chunk = []
                current_len = 0

            sub_chunks = _split_text(split, new_separators, max_chars, overlap_chars)
            chunks.extend(sub_chunks)
            continue

        # ------------------------------------------------------------------
        # Normal accumulation path.
        # ------------------------------------------------------------------
        split_len = len(split) + (len(separator_str) if current_chunk else 0)

        if current_len + split_len > max_chars and current_chunk:
            chunks.append(separator_str.join(current_chunk))

            # Compute overlap: walk backwards through current_chunk keeping
            # whole units until we would exceed overlap_chars.
            overlap_chunk = []
            overlap_len = 0
            for w in reversed(current_chunk):
                w_len = len(w) + (len(separator_str) if overlap_chunk else 0)
                if overlap_len + w_len > overlap_chars:
                    break
                overlap_chunk.insert(0, w)
                overlap_len += w_len

            if not overlap_chunk:
                # Every unit exceeds overlap_chars (e.g. long sentences at
                # the sentence-split level). Fall back to a char-level suffix
                # of the last unit so the boundary is not completely cold.
                tail = current_chunk[-1][-overlap_chars:]
                overlap_chunk = [tail]
                overlap_len = len(tail)

            current_chunk = overlap_chunk
            current_len = overlap_len
            split_len = len(split) + (len(separator_str) if current_chunk else 0)

        current_chunk.append(split)
        current_len += split_len

    if current_chunk:
        chunks.append(separator_str.join(current_chunk))

    return chunks


def measure_overlap_accuracy(chunks, target_overlap_chars=DEFAULT_OVERLAP_CHARS):
    """
    Measures the ACTUAL character overlap between consecutive chunks and
    compares it against the configured target. Originally built to
    quantify the overlap-math discrepancy identified during code review;
    now also serves as the ongoing regression check proving the v2 fix
    keeps real overlap at or under target_overlap_chars.

    Overlap is found by locating the longest suffix of chunk[i] that
    exactly matches a prefix of chunk[i+1] - exact match works here because
    chunk_text literally re-inserts the same overlap words verbatim.
    """
    if len(chunks) < 2:
        return {
            "avg_overlap_chars": 0,
            "max_overlap_chars": 0,
            "target_overlap_chars": target_overlap_chars,
        }

    overlaps = []
    for a, b in zip(chunks, chunks[1:]):
        max_check = min(len(a), len(b), target_overlap_chars * 3)
        found = 0
        for length in range(max_check, 0, -1):
            if a[-length:] == b[:length]:
                found = length
                break
        overlaps.append(found)

    return {
        "avg_overlap_chars": round(sum(overlaps) / len(overlaps), 1) if overlaps else 0,
        "max_overlap_chars": max(overlaps) if overlaps else 0,
        "target_overlap_chars": target_overlap_chars,
    }