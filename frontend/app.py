import io
import requests
import streamlit as st
from PIL import Image

# Page setup
st.set_page_config(
    page_title="Document Intelligence RAG", page_icon="🏗️", layout="wide"
)

# Sidebar Configuration
st.sidebar.header("⚙️ Settings")
API_BASE_URL = st.sidebar.text_input(
    "FastAPI Base URL", value="http://localhost:8000"
)

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

# 2. Chunk Preview Inspector
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Chunk Inspector")
limit = st.sidebar.slider("Chunk Limit", min_value=1, max_value=20, value=5)
chars = st.sidebar.slider(
    "Chars Per Chunk", min_value=100, max_value=1000, value=400
)

if st.sidebar.button("Preview Chunks", use_container_width=True):
    try:
        res = requests.post(
            f"{API_BASE_URL}/preview-chunks",
            json={"limit": limit, "chars": chars},
            timeout=10,
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


# --- MODAL DIALOG FOR CITATION HIGHLIGHT PREVIEW ---
@st.dialog("📄 Citation Highlight Viewer", width="large")
def view_citation_modal(source_item: dict):
    doc_name = source_item.get("source", "Unknown PDF")
    page_no = source_item.get("page_number", 1)
    citation_id = source_item.get("citation_id", 1)

    st.subheader(f"Source {citation_id}: {doc_name}")
    st.caption(f"Page Number: {page_no}")

    with st.spinner("Rendering document bounding box..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/render-highlight",
                json={
                    "source": doc_name,
                    "page_number": page_no,
                    "dl_prov": source_item.get("dl_prov", []),
                },
                timeout=15,
            )
            if response.status_code == 200:
                image_bytes = response.content
                image = Image.open(io.BytesIO(image_bytes))
                st.image(image, use_container_width=True)
            else:
                st.error(
                    f"Failed to render image ({response.status_code}): {response.text}"
                )
        except Exception as err:
            st.error(f"Could not connect to render endpoint: {err}")


# --- MAIN UI: Chat Interface ---
st.title("🏗️ Document Intelligence Assistant")
st.caption(
    "Ask questions about your uploaded construction plans, specs, and"
    " documents."
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_citations(sources: list, msg_idx: int):
    """Renders interactive citation buttons that open pop-up modals."""
    st.markdown("---")
    st.markdown("**Cited Sources:**")
    cols = st.columns(min(len(sources), 5))

    for idx, src in enumerate(sources):
        col_idx = idx % 5
        citation_id = src.get("citation_id", idx + 1)
        btn_label = f"📌 [Source {citation_id}]"

        if cols[col_idx].button(
            btn_label, key=f"cite_btn_{msg_idx}_{citation_id}"
        ):
            view_citation_modal(src)


# Render chat history
for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_citations(message["sources"], msg_idx)

# Handle user input
if prompt := st.chat_input("Ask a question about your project docs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔍 Searching vector store...")

        try:
            res = requests.post(
                f"{API_BASE_URL}/ask", json={"question": prompt}, timeout=60
            )

            if res.status_code == 200:
                data = res.json()
                answer = data.get("answer", "No response received.")
                sources = data.get("sources", [])

                message_placeholder.markdown(answer)

                assistant_msg_idx = len(st.session_state.messages)
                if sources:
                    render_citations(sources, assistant_msg_idx)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            else:
                message_placeholder.error(
                    f"API Error ({res.status_code}): {res.text}"
                )

        except Exception as e:
            message_placeholder.error(
                f"Failed to connect to backend at {API_BASE_URL}: {e}"
            )