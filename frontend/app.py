import io
import os
import requests
import streamlit as st
from PIL import Image
import google.auth.transport.requests
import google.oauth2.id_token

st.set_page_config(
    page_title="Document Intelligence RAG", page_icon="🏗️", layout="wide"
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

API_BASE_URL = st.sidebar.text_input(
    "FastAPI Base URL", value=BACKEND_URL
).rstrip("/")


def get_auth_headers(target_url: str) -> dict:
    """Generates an OIDC ID Token for Cloud Run service-to-service authentication."""
    if "localhost" in target_url or "127.0.0.1" in target_url:
        return {}

    try:
        auth_req = google.auth.transport.requests.Request()
        token = google.oauth2.id_token.fetch_id_token(auth_req, target_url)
        return {"Authorization": f"Bearer {token}"}
    except Exception as err:
        st.error(f"Authentication token generation failed: {err}")
        return {}


try:
    health_res = requests.get(
        f"{API_BASE_URL}/health",
        headers=get_auth_headers(API_BASE_URL),
        timeout=15,
    )
    if health_res.status_code == 200:
        st.sidebar.success("Backend Connected 🟢")
    else:
        st.sidebar.warning("Backend Issue 🟡")
except Exception:
    st.sidebar.error("Backend Offline 🔴")

st.sidebar.markdown("---")

properties = []
try:
    prop_res = requests.get(
        f"{API_BASE_URL}/properties",
        headers=get_auth_headers(API_BASE_URL),
        timeout=5,
    )
    if prop_res.status_code == 200:
        properties = prop_res.json().get("properties", [])
except Exception:
    pass

selected_property = st.sidebar.selectbox(
    "🏢 Select Property Context",
    options=["All"] + properties,
)

st.sidebar.markdown("---")

st.sidebar.subheader("📤 Upload & Ingest Documents")

if properties:
    upload_mode = st.sidebar.radio(
        "Upload Target",
        options=["Existing Property", "➕ New Property"],
        horizontal=True,
    )

    if upload_mode == "Existing Property":
        target_property = st.sidebar.selectbox(
            "Select Existing Property",
            options=properties,
            key="upload_existing_prop_select",
        )
    else:
        target_property = st.sidebar.text_input(
            "New Property Name",
            placeholder="e.g., Oak_Ridge_Site",
            key="upload_new_prop_input",
        )
else:
    target_property = st.sidebar.text_input(
        "Property Name",
        placeholder="e.g., Oak_Ridge_Site",
        key="upload_new_prop_input",
    )

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    accept_multiple_files=True,
    type=["pdf"],
)

if st.sidebar.button("Ingest Files", use_container_width=True):
    clean_target_property = target_property.strip().replace(" ", "_") if target_property else ""
    
    if clean_target_property and uploaded_files:
        files_payload = [
            ("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files
        ]
        with st.sidebar.spinner("Uploading, parsing, & indexing..."):
            try:
                res = requests.post(
                    f"{API_BASE_URL}/upload-and-ingest",
                    data={"property_name": clean_target_property},
                    files=files_payload,
                    headers=get_auth_headers(API_BASE_URL),
                    timeout=180,
                )
                if res.status_code == 200:
                    st.sidebar.success(f"Ingested to '{clean_target_property}'!")
                    st.rerun()
                else:
                    st.sidebar.error(f"Ingestion failed ({res.status_code}): {res.text}")
            except Exception as e:
                st.sidebar.error(f"Failed to connect: {e}")
    else:
        st.sidebar.warning("Please provide a valid property name and select at least one PDF.")


@st.dialog("📄 Citation Highlight Viewer", width="large")
def view_citation_modal(source_item: dict):
    """Display the highlighted PDF page associated with a source citation."""
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
                headers=get_auth_headers(API_BASE_URL),
                timeout=15,
            )
            if response.status_code == 200:
                image_bytes = response.content
                image = Image.open(io.BytesIO(image_bytes))
                st.image(image, output_format="PNG")
            else:
                st.error(
                    f"Failed to render image ({response.status_code}): {response.text}"
                )
        except Exception as err:
            st.error(f"Could not connect to render endpoint: {err}")

st.title("🏗️ Document Intelligence Assistant")
st.caption(
    "Ask questions about your uploaded construction plans, specs, and documents."
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_citations(sources: list, msg_idx: int):
    """Render clickable citation buttons that open the matching highlight popup."""
    st.markdown("---")
    st.markdown("**Cited Sources:**")
    cols = st.columns(min(len(sources), 5))

    for idx, src in enumerate(sources):
        col_idx = idx % 5
        citation_id = src.get("citation_id", idx + 1)
        btn_label = f"📌 [Source {citation_id}]"

        if cols[col_idx].button(
            btn_label,
            key=f"cite_btn_{msg_idx}_{citation_id}",
        ):
            view_citation_modal(src)


for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_citations(message["sources"], msg_idx)

if prompt := st.chat_input("Ask a question about your project docs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔍 Searching vector store...")

        try:
            res = requests.post(
                f"{API_BASE_URL}/ask",
                json={"question": prompt, "property_name": selected_property},
                headers=get_auth_headers(API_BASE_URL),
                timeout=60,
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