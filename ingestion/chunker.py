"""
chunker.py

Responsible for splitting raw text into retrieval-ready chunks.

This module performs no embedding, storage, or document reading.
It simply converts text into overlapping chunks that are later
embedded and indexed.
"""

VERSION = "word-overlap-v1"
DEFAULT_MAX_CHARS = 1500
DEFAULT_OVERLAP_CHARS = 150
 

def chunk_text(text, max_chars=1500, overlap_chars=150):
    """
    Splits text into chunks bounded by character length, not word count.
    Falls back to hard character-slicing for any single 'word' that alone
    exceeds max_chars (e.g. CSV rows, DNA sequences, unbroken log lines).
    """

    words = text.split()
    chunks=[]
    current_chunk=[]
    current_length=0

    for word in words:
        if len(word) > max_chars:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk=[]
                current_length=0
            for i in range(0,len(word),max_chars - overlap_chars):
                chunks.append(word[i:i + max_chars])
            continue

        if current_length +len(word) +1>max_chars:
            chunks.append(" ".join(current_chunk))
            overlap_words=[]
            overlap_len=0
            for w in reversed(current_chunk):
                if overlap_len - len(w) > overlap_chars:
                    break
                overlap_words.insert(0,w)
                overlap_len += len(w)+1
            current_chunk=overlap_words
            current_length = overlap_len

        current_chunk.append(word)
        current_length += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks



def measure_overlap_accuracy(chunks, target_overlap_chars=DEFAULT_OVERLAP_CHARS):
    """
    Measures the ACTUAL character overlap between consecutive chunks and
    compares it against the configured target. This exists specifically to
    quantify the overlap-math discrepancy identified during code review
    (chunk_text's boundary check can overshoot target_overlap_chars) -
    turning a suspected bug into a tracked, per-run number instead of a
    one-time observation.
 
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
        "avg_overlap_chars": round(sum(overlaps) / len(overlaps), 1),
        "max_overlap_chars": max(overlaps),
        "target_overlap_chars": target_overlap_chars,
    }
 
