"""
chunker.py

Responsible for splitting raw text into retrieval-ready chunks.

This module performs no embedding, storage, or document reading.
It simply converts text into overlapping chunks that are later
embedded and indexed.
"""


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


