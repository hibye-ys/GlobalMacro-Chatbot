import yaml
import json
from datetime import datetime
import streamlit as st
from langchain_core.messages.chat import ChatMessage


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_chat_history_json(chat_history, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json_data = [
            {"role": message.role, "content": message.content}
            for message in chat_history
        ]
        json.dump(json_data, f)


def load_chat_history_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
        messages = [
            ChatMessage(role=message["role"], content=message["content"])
            for message in json_data
        ]
        return messages


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))
