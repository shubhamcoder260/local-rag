import os
import sys
import requests

API_BASE_URL = "http://127.0.0.1:8000"

def ingest_text_file(file_path):
    """Read a .txt file and sends its content to the /ingest endpoint."""
    with open(file_path, "r", encoding="utf-8") as f:
        text_content = f.read()

    source_name = os.path.basename(file_path)

    payload ={
        "text_content": text_content,
        "source_name": source_name
    }

    response = requests.post(f"{API_BASE_URL}/ingest",json=payload)

    if response.status_code ==200:
        print(f"✅ {source_name}: {response.json()['message']}")
    else:
        print(f"❌ {source_name}: FAILED ({response.status_code}) - {response.text}")


def ingest_pdf_file(file_path):
    """Sends a .pdf file to the /ingest/pdf endpoint as a file upload."""
    source_name = os.path.basename(file_path)

    with open(file_path, "rb") as f:
        files = {"file": (source_name, f,"application/pdf")}
        response = requests.post(f"{API_BASE_URL}/ingest/pdf", files=files)

    if response.status_code ==200:
        print(f"✅ {source_name}: {response.json()['message']}")
    else:
        print(f"❌ {source_name}: FAILED ({response.status_code}) - {response.text}")


if __name__=="__main__":
    if len(sys.argv) <2:
        print("Usage: python3 bulk_ingest.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]

    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid folder.")
        sys.exit(1)

    files = os.listdir(folder_path)
    print(f"Found {len(files)} item(s) in '{folder_path}'\n")

    for filename in files:
        full_path = os.path.join(folder_path,filename)

        if not os.path.isfile(full_path):
            continue

        if filename.lower().endswith(".txt"):
            ingest_text_file(full_path)
        elif filename.lower().endswith(".pdf"):
            ingest_pdf_file(full_path)
        else:
            print(f"⏭️  {filename}: skipped (unsupported file type)")

    print("\nBulk ingestion complete.")