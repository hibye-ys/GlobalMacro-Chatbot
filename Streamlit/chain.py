from langchain_core.runnables import RunnablePassthrough
from retriever import (
    get_bm25_retriever,
    get_multivector_retriever,
    get_parentdocument_retriever,
)
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import os
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from utils import load_yaml
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import ChatMessageHistory


def create_chain(model_name, retriever, prompt, chat_history):

    # load_dotenv()
    # logging.langsmith("streamlit_test")

    if retriever == "BM25":
        retriever = get_bm25_retriever()
    elif retriever == "MultiVector":
        retriever = get_multivector_retriever()
    elif retriever == "ParentDocument":
        retriever = get_parentdocument_retriever()

    message_history = ChatMessageHistory()
    for message in chat_history:
        message_history.add_message(message)

    memory = ConversationBufferWindowMemory(
        memory_key="history", chat_memory=message_history, k=2, return_messages=True
    )
    prompt_template = load_yaml(prompt)["prompt"]
    prompt = PromptTemplate.from_template(
        prompt_template,
        partial_variables={"history": memory.load_memory_variables({})["history"]},
    )

    if model_name.startswith("claude"):
        llm = ChatAnthropic(model_name=model_name)
    else:
        llm = ChatOpenAI(model_name=model_name)

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
