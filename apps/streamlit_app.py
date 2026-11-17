"""Chat UI for the agent API, with live node-by-node streaming and an Approve/Reject step.

    uvicorn ailearn.api:app --app-dir src          # terminal 1
    streamlit run apps/streamlit_app.py            # terminal 2  (API_URL defaults to localhost:8000)
"""
import json
import os
import uuid

import httpx
import streamlit as st

API = os.getenv("API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="Data Platform Agent", page_icon="🧱")
st.title("🧱 Data Platform Agent")

if "thread_id" not in st.session_state:
    st.session_state.thread_id, st.session_state.history, st.session_state.pending = str(uuid.uuid4()), [], None

for role, text in st.session_state.history:
    st.chat_message(role).markdown(text)


def stream_ask(question: str) -> dict:
    """Consume Server-Sent Events from /ask/stream, showing each graph node as it finishes."""
    result, status = None, st.status("Thinking…", expanded=True)
    with httpx.stream("POST", f"{API}/ask/stream", json={"question": question,
                      "thread_id": st.session_state.thread_id}, timeout=120) as r:
        event = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if event == "node":
                    status.write(f"✅ **{data['node']}** {(' — ' + data['trace'][0]) if data['trace'] else ''}")
                elif event == "result":
                    result = data
    status.update(label="Done", state="complete", expanded=False)
    return result


if q := st.chat_input("Ask about data, docs, or request an action…"):
    st.session_state.history.append(("user", q))
    st.chat_message("user").markdown(q)
    res = stream_ask(q)
    if res["status"] == "pending_approval":
        st.session_state.pending = res["pending_action"]
    else:
        st.session_state.history.append(("assistant", res["answer"]))
        st.chat_message("assistant").markdown(res["answer"])

if st.session_state.pending:
    st.warning(f"⏸ Approval needed: **{st.session_state.pending}**")
    c1, c2 = st.columns(2)
    for col, label, ok in ((c1, "Approve", True), (c2, "Reject", False)):
        if col.button(label, use_container_width=True):
            res = httpx.post(f"{API}/approve/{st.session_state.thread_id}", json={"approved": ok}, timeout=60).json()
            st.session_state.history.append(("assistant", res["answer"]))
            st.session_state.pending = None
            st.session_state.thread_id = str(uuid.uuid4())   # new thread for the next request
            st.rerun()
