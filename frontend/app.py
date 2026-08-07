import streamlit as st
import requests

# Page setup
st.set_page_config(page_title="Document Intelligence RAG", page_icon="🏗️", layout="wide")

# Sidebar Configuration
st.sidebar.header("⚙️ Settings")
API_BASE_URL = st.sidebar.text_input("FastAPI Base URL", value="http://localhost:8000")

# Health Check Indicator
try:
    health_res = requests.get(f"{API_BASE_URL}/health", timeout=3)
    if health_res.status_code == 200:
        st.sidebar.success("Backend Connected 🟢")
    else:
        st.sidebar.warning("Backend Issue 🟡")
except Exception:
    st.sidebar.error("Backend Offline 🔴")

st.sidebar.markdown("---")

# 1. Ingestion Control
st.sidebar.subheader("📄 Document Ingestion")
if st.sidebar.button("🚀 Trigger Ingestion", use_container_width=True):
    with st.sidebar.status("Ingesting documents...", expanded=True) as status:
        try:
            res = requests.post(f"{API_BASE_URL}/ingest", timeout=600)
            if res.status_code == 200:
                data = res.json()
                status.update(label="✅ Ingestion Complete!", state="complete", expanded=False)
                st.sidebar.success(data.get("message", "Success!"))
                if "chunks_indexed" in data:
                    st.sidebar.info(f"Indexed {data['chunks_indexed']} chunks.")
            else:
                status.update(label="❌ Ingestion Failed", state="error")
                st.sidebar.error(f"Error ({res.status_code}): {res.text}")
        except Exception as e:
            status.update(label="❌ Error Connecting", state="error")
            st.sidebar.error(f"Request failed: {e}")

# 2. Chunk Preview Inspector
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Chunk Inspector")
limit = st.sidebar.slider("Chunk Limit", min_value=1, max_value=20, value=5)
chars = st.sidebar.slider("Chars Per Chunk", min_value=100, max_value=1000, value=400)

if st.sidebar.button("Preview Chunks", use_container_width=True):
    try:
        res = requests.post(
            f"{API_BASE_URL}/preview-chunks",
            json={"limit": limit, "chars": chars},
            timeout=10
        )
        if res.status_code == 200:
            chunks = res.json().get("chunks", [])
            st.session_state["preview_chunks"] = chunks
        else:
            st.sidebar.error(f"Error ({res.status_code}): {res.text}")
    except Exception as e:
        st.sidebar.error(f"Preview failed: {e}")

if "preview_chunks" in st.session_state:
    with st.sidebar.expander("Inspected Chunk Heads", expanded=False):
        for idx, chunk in enumerate(st.session_state["preview_chunks"], 1):
            st.markdown(f"**Chunk {idx}:**")
            st.code(chunk, language="text")

# --- MAIN UI: Chat Interface ---
st.title("🏗️ Document Intelligence Assistant")
st.caption("Ask questions about your uploaded construction plans, specs, and documents.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 View Citations"):
                for idx, src in enumerate(message["sources"], 1):
                    source_file = src.get("source", "Unknown")
                    page = src.get("page_number", src.get("page_numbers", "N/A"))
                    st.write(f"**{idx}. {source_file}** — *Page {page}*")

# Handle user input
if prompt := st.chat_input("Ask a question about your project docs..."):
    # Append & display user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI `/ask` endpoint
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔍 Searching vector store...")

        try:
            res = requests.post(
                f"{API_BASE_URL}/ask",
                json={"question": prompt},
                timeout=60
            )

            if res.status_code == 200:
                data = res.json()
                answer = data.get("answer", "No response received.")
                sources = data.get("sources", [])

                message_placeholder.markdown(answer)

                if sources:
                    with st.expander("📚 View Citations"):
                        for idx, src in enumerate(sources, 1):
                            source_file = src.get("source", "Unknown")
                            page = src.get("page_number", src.get("page_numbers", "N/A"))
                            st.write(f"**{idx}. {source_file}** — *Page {page}*")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            else:
                message_placeholder.error(f"API Error ({res.status_code}): {res.text}")

        except Exception as e:
            message_placeholder.error(f"Failed to connect to backend at {API_BASE_URL}: {e}")