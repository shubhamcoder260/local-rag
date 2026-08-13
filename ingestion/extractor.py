import fitz

VERSION = "pymupdf-v1"

def extract_text_from_pdf(file_bytes):
    """Opens a PDF from raw bytes and extracts all text, page by page."""
    with fitz.open(stream=file_bytes,filetype="pdf") as doc:
        full_text=""
        for page in doc:
            full_text+= page.get_text()
    return full_text
