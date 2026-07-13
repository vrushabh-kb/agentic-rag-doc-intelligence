import uuid
import requests
import streamlit as st

st.set_page_config(page_title="Document Intelligence Agent", page_icon="📄")

try:
    API_URL = st.secrets.get("API_URL", "http://localhost:8000")
except FileNotFoundError:
    API_URL = "http://localhost:8000"

st.title("📄 Agentic Document Intelligence")
st.caption("Multi-document Q&A with deterministic routing + session memory")
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask a question about your documents...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Routing and retrieving..."):
            resp = requests.post(
                f"{API_URL}/chat",
                json={"session_id": st.session_state.session_id, "query": query},
                timeout=60,
            )
            data = resp.json()
        st.write(data["answer"])

        with st.expander("🔍 Routing details (for demo/debug)"):
            st.write(f"**Query type:** {data['query_type']}")
            st.write(f"**Routing reason:** {data['routing_reason']}")
            st.write(f"**Docs searched:** {data['docs_searched'] or 'all documents'}")
            for s in data["sources"]:
                st.write(f"- {s['doc_title']} (page {s['page_number']})")

    st.session_state.messages.append({"role": "assistant", "content": data["answer"]})
