import streamlit as st
import requests
import json
import time
import csv
import os
from datetime import datetime

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

def log_benchmark(question, answer, context, model, document, elapsed_seconds):
    """Appends one benchmark entry to a CSV log file."""
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/benchmark_log.csv"
    file_exists = os.path.isfile(log_file)

    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "question", "model", "document",
                "time_seconds", "answer_length_chars", "context_length_chars",
                "answer", "context"
            ])
        writer.writerow([
            datetime.now().isoformat(),
            question,
            model,
            document,
            round(elapsed_seconds, 2),
            len(answer),
            len(context),
            answer,
            context
        ])

st.set_page_config(page_title="Local RAG", layout="wide")
st.title("🧠 Local RAG Knowledge Base")

# --- Sidebar: Ingest ---
with st.sidebar:
    st.header("Upload a Document")
    uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])

    if uploaded_file is not None:
        if st.button("Ingest File"):
            with st.spinner("Ingesting..."):
                if uploaded_file.name.lower().endswith(".pdf"):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{API_BASE_URL}/ingest/pdf", files=files)
                else:
                    text_content = uploaded_file.getvalue().decode("utf-8")
                    payload = {"text_content": text_content, "source_name": uploaded_file.name}
                    response = requests.post(f"{API_BASE_URL}/ingest", json=payload)

                if response.status_code == 200:
                    st.success(response.json()["message"])
                else:
                    st.error(f"Failed: {response.text}")
    st.divider()
    if os.path.exists("logs/benchmark_log.csv"):
        with open("logs/benchmark_log.csv", "rb") as f:
            st.download_button(
                label="Download benchmark log",
                data=f,
                file_name="benchmark_log.csv",
                mime="text/csv"

            )
    st.divider()
    st.header("Privacy")
    record_data = st.checkbox("TESTER MODE" , value=False)
    st.caption("When enabled, your questions, and retrived context are saved locally to benchmark_log.csv. Nothing leaves your machine.")
# --- Main: Ask ---
st.header("Ask a Question")

docs_response = requests.get(f"{API_BASE_URL}/documents")
available_docs = docs_response.json()["documents"] if docs_response.status_code == 200 else []

selected_doc = st.selectbox("Choose a document to search:", available_docs)
models_response=requests.get(f"{API_BASE_URL}/models")
all_models = models_response.json()["models"] if models_response.status_code ==200 else []
chat_models = [m for m in all_models if "embed" not in m.lower()]

selected_model = st.selectbox("Choose a model:", chat_models)  



question = st.text_input("Your question:")

if st.button("Ask"):
    if not selected_doc:
        st.warning("No documents available — ingest one first.")

    elif not selected_model:
        st.warning("No models available")

    elif not question:
        st.warning("Type a question first.")
    else:
        payload = {"question": question, "source_collection": selected_doc, "llm_model": selected_model}

        start_time = time.time()

        with st.spinner("Thinking... this may take a while depending on your model and hardware."):
            with requests.post(f"{API_BASE_URL}/ask", json=payload, stream=True) as response:
                if response.status_code != 200:
                    st.error(f"Failed: {response.text}")
                else:
                    lines = response.iter_lines(decode_unicode=True)
                    metadata_line = next(lines)
                    metadata = json.loads(metadata_line)

                    st.markdown("### Answer")
                    answer_placeholder = st.empty()
                    full_answer = ""
                    for line in lines:
                        full_answer += line
                        answer_placeholder.write(full_answer)

                    elapsed = time.time() - start_time
                    st.caption(f"Answered in {elapsed:.1f} seconds")
                    if record_data:
                        log_benchmark(question, full_answer, metadata["context_used"], selected_model, selected_doc, elapsed)

                    with st.expander("Show retrieved context"):
                        st.text(metadata["context_used"])