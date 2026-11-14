"""Streamlit frontend for Contract Document Chat.

Run with:  streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
POLL_SECONDS = 1.0
INDEX_TIMEOUT_SECONDS = 600

STATUS_BADGES = {
    "ready": "🟢 ready",
    "processing": "🟡 processing",
    "not_indexed": "⚪ not indexed",
    "failed": "🔴 failed",
}

st.set_page_config(page_title="Contract Chat", page_icon="📄", layout="wide")


# --------------------------------------------------------------------------- #
# Backend calls
# --------------------------------------------------------------------------- #


def api(method: str, path: str, **kwargs: Any) -> Any:
    """Call the backend and return JSON; raises RuntimeError with a readable message."""
    try:
        resp = requests.request(method, f"{BACKEND_URL}{path}", timeout=180, **kwargs)
    except requests.RequestException as exc:
        raise RuntimeError(f"Backend not reachable at {BACKEND_URL}: {exc}") from exc
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise RuntimeError(f"{resp.status_code}: {detail}")
    return resp.json()


def index_and_wait(doc_id: str, force: bool = False) -> dict[str, Any]:
    """Trigger indexing and poll until the document is ready or failed."""
    doc = api("POST", f"/documents/{doc_id}/index", params={"force": str(force).lower()})
    start = time.time()
    while doc["status"] in ("processing", "not_indexed"):
        if time.time() - start > INDEX_TIMEOUT_SECONDS:
            raise RuntimeError("Timed out waiting for indexing")
        time.sleep(POLL_SECONDS)
        doc = api("GET", f"/documents/{doc_id}")
    return doc


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

st.session_state.setdefault("selected", None)  # dict of the selected document
st.session_state.setdefault("messages", [])  # [{role, content, citations?}]


def select_document(doc: dict[str, Any], force: bool = False) -> None:
    """Index (if needed) and switch the chat to ``doc``; clears the chat history."""
    with st.spinner(f"Preparing **{doc['file_name']}** (embedding if not indexed yet)…"):
        try:
            final = index_and_wait(doc["doc_id"], force=force)
        except RuntimeError as exc:
            st.sidebar.error(str(exc))
            return
    if final["status"] == "ready":
        if not st.session_state.selected or st.session_state.selected["doc_id"] != final["doc_id"] or force:
            st.session_state.messages = []
        st.session_state.selected = final
    else:
        st.sidebar.error(f"Indexing failed: {final.get('error') or 'unknown error'}")


# --------------------------------------------------------------------------- #
# Sidebar: document list
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("📁 Contracts")
    if st.button("🔄 Refresh list", use_container_width=True):
        st.rerun()
    try:
        documents = api("GET", "/documents")
    except RuntimeError as exc:
        st.error(str(exc))
        documents = []

    if not documents:
        st.info("No .pdf, .docx or .txt files found in the contracts folder.")

    for doc in documents:
        is_selected = bool(st.session_state.selected and st.session_state.selected["doc_id"] == doc["doc_id"])
        with st.container(border=True):
            st.markdown(f"**{'▶ ' if is_selected else ''}{doc['file_name']}**")
            meta = f"{STATUS_BADGES.get(doc['status'], doc['status'])} · {human_size(doc['size'])}"
            if doc.get("num_chunks"):
                meta += f" · {doc['num_chunks']} chunks"
            st.caption(meta)
            if doc["status"] == "failed" and doc.get("error"):
                st.caption(f"⚠️ {doc['error']}")
            c1, c2 = st.columns(2)
            if c1.button("Select", key=f"sel-{doc['doc_id']}", type="primary" if not is_selected else "secondary",
                         use_container_width=True):
                select_document(doc)
                st.rerun()
            if doc["status"] in ("ready", "failed") and c2.button(
                "Re-index", key=f"re-{doc['doc_id']}", use_container_width=True
            ):
                select_document(doc, force=True)
                st.rerun()

# --------------------------------------------------------------------------- #
# Main area: chat
# --------------------------------------------------------------------------- #

selected = st.session_state.selected
if selected and documents and selected["doc_id"] not in {d["doc_id"] for d in documents}:
    st.warning("The selected contract changed on disk or was removed. Please select it again.")
    st.session_state.selected, st.session_state.messages = None, []
    selected = None

if not selected:
    st.title("📄 Contract Chat")
    st.write("Select a contract in the sidebar to start chatting. Answers are grounded only in that contract.")
    st.stop()

st.title(f"📄 {selected['file_name']}")
st.caption("Answers use only this contract and cite clause numbers. Switching contracts clears the chat.")


def render_citations(citations: list[dict[str, Any]]) -> None:
    if not citations:
        return
    with st.expander(f"📎 Citations ({len(citations)})"):
        for c in citations:
            title = f" – {c['clause_title']}" if c.get("clause_title") else ""
            page = f" (p. {c['page']})" if c.get("page") else ""
            label = c["clause_no"] if not c["clause_no"][:1].isdigit() else f"Clause {c['clause_no']}"
            st.markdown(f"**{label}{title}**{page}")
            st.markdown(f"> {c['snippet']}".replace("\n", "\n> "))


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_citations(msg.get("citations", []))

if question := st.chat_input("Ask a question about this contract…"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages
               if not m.get("error")]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the contract…"):
            try:
                result = api("POST", "/chat", json={"doc_id": selected["doc_id"], "question": question,
                                                     "history": history})
                answer, citations, failed = result["answer"], result.get("citations", []), False
            except RuntimeError as exc:
                answer, citations, failed = f"⚠️ {exc}", [], True
        st.markdown(answer)
        render_citations(citations)
    if failed:  # keep failed turns out of the history sent to the backend
        st.session_state.messages[-1]["error"] = True
    st.session_state.messages.append({"role": "assistant", "content": answer, "citations": citations,
                                      "error": failed})
