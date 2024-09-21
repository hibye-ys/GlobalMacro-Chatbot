import streamlit as st


def init_state():
    if "session_key" not in st.session_state:
        st.session_state["session_key"] = "new_session"

    if "session_index_tracker" not in st.session_state:
        st.session_state["session_index_tracker"] = "new_session"

    if "new_session_key" not in st.session_state:
        st.session_state["new_session_key"] = None

    if (
        st.session_state["session_key"] == "new_session"
        and st.session_state["new_session_key"] != None
    ):
        st.session_state["session_index_tracker"] = st.session_state["new_session_key"]
        st.session_state["new_session_key"] = None

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "store" not in st.session_state:
        st.session_state["store"] = {}
