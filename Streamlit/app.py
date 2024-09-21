import streamlit as st
from dotenv import load_dotenv
from utils import (
    load_yaml,
    load_chat_history_json,
    save_chat_history_json,
    print_messages,
    add_message,
)
from chain import create_chain
import glob

import os
from init_state import init_state
from datetime import datetime

# from langchain_teddynote import logging


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def save_chat_history():
    if st.session_state.messages != []:
        if st.session_state["session_key"] == "new_session":
            st.session_state.new_session_key = get_timestamp() + ".json"
            session_path = f"./chat_sessions/{st.session_state.new_session_key}"
            save_chat_history_json(st.session_state.messages, session_path)
        else:
            session_path = f"./chat_sessions/{st.session_state.session_key}"
            save_chat_history_json(st.session_state.messages, session_path)


def track_index():
    st.session_state.session_index_tracker = st.session_state.session_key


def main():
    load_dotenv()
    # logging.langsmith("streamlit_test")
    st.title("GlobalMacro ChatBot 💬")
    init_state()
    chat_sessions = ["new_session"] + os.listdir("./chat_sessions")

    with st.sidebar:
        index = chat_sessions.index(st.session_state["session_index_tracker"])
        st.selectbox(
            "select session",
            chat_sessions,
            key="session_key",
            index=index,
            on_change=track_index,
        )

        prompt_files = glob.glob("./prompts/*.yaml")
        selected_prompt = st.selectbox("select prompt", prompt_files, index=0)

        selected_model = st.selectbox(
            "select model",
            ["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20240620"],
            index=0,
        )

        selected_retriver = st.selectbox(
            "select retriever", ["BM25", "MultiVector", "ParentDocument"], index=0
        )

    if st.session_state["session_key"] != "new_session":
        st.session_state.messages = load_chat_history_json(
            f"./chat_sessions/{st.session_state.session_key}"
        )
    else:
        st.session_state.messages = []

    print_messages()

    chain = create_chain(
        selected_model, selected_retriver, selected_prompt, st.session_state.messages
    )
    user_input = st.chat_input("궁금한 내용을 물어보세요!", key="user_input")

    if user_input:

        st.chat_message("user").write(user_input)

        response = chain.stream(user_input)

        with st.chat_message("assistant"):
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        add_message("user", user_input)
        add_message("assistant", ai_answer)

        save_chat_history()


if __name__ == "__main__":

    main()
