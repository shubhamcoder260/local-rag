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

    # --- Benchmarking ---
    st.header("Benchmarking")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ingestion")

        ingestion_enabled = st.checkbox(
            "Record ingestion benchmark",
            value=True,
            key="ingestion_benchmark_toggle",
        )

        # Push the toggle to the API only when it actually changes - avoids
        # spamming the config endpoint on every Streamlit rerun.
        if st.session_state.get("last_ingestion_toggle") != ingestion_enabled:
            try:
                requests.post(
                    f"{API_BASE_URL}/benchmark/config",
                    json={"ingestion_enabled": ingestion_enabled},
                    timeout=5,
                )
                st.session_state["last_ingestion_toggle"] = ingestion_enabled
            except Exception as e:
                st.warning(f"Could not update ingestion benchmark setting: {e}")

        if st.button("Download Ingestion Benchmark"):
            try:
                resp = requests.get(f"{API_BASE_URL}/benchmark/ingestion", timeout=10)
                if resp.status_code == 200:
                    st.download_button(
                        label="Save ingestion_benchmark.csv",
                        data=resp.content,
                        file_name="ingestion_benchmark.csv",
                        mime="text/csv",
                        key="download_ingestion_csv",
                    )
                else:
                    st.warning("No ingestion benchmark data available yet.")
            except Exception as e:
                st.error(f"Failed to fetch ingestion benchmark: {e}")

    with col2:
        st.subheader("Retrieval")

        record_data = st.checkbox(
            "Record retrieval benchmark (TESTER MODE)",
            value=False,
            key="retrieval_benchmark_toggle",
        )
        st.caption(
            "When enabled, your questions and retrieved context are saved "
            "locally to benchmark_log.csv. Nothing leaves your machine."
        )

        if st.button("Download Retrieval Benchmark"):
            try:
                with open("logs/benchmark_log.csv", "rb") as f:
                    st.download_button(
                        label="Save retrieval_benchmark.csv",
                        data=f.read(),
                        file_name="retrieval_benchmark.csv",
                        mime="text/csv",
                        key="download_retrieval_csv",
                    )
            except FileNotFoundError:
                st.warning("No retrieval benchmark data available yet.")

st.divider()

st.subheader("Knowledge Base")

if st.button(" Clear Knowledge Base", type="primary"):

    response = requests.delete(f"{API_BASE_URL}/knowledgebase")

    if response.status_code ==200:
        st.success(response.json()["message"])
        st.rerun()
    else:
        st.error(response.text)
    


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