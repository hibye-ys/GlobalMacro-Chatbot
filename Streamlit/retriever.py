from dotenv import load_dotenv
from langchain_teddynote.community.pinecone import init_pinecone_index
from langchain_upstage.embeddings import UpstageEmbeddings
from langchain_teddynote.community.pinecone import PineconeKiwiHybridRetriever
from langchain_teddynote.korean import stopwords
import os
import uuid
from langchain_chroma import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_openai import ChatOpenAI
from langchain_upstage.embeddings import UpstageEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.storage.encoder_backed import EncoderBackedStore
import pickle
from langchain_community.vectorstores.utils import filter_complex_metadata
import json
import sys
import os

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from DataProcessing.utils import load_yaml
from DataProcessing.extract_graphstate import (
    extract_documents_for_multivectorstore,
    extract_documents_for_multidocstore,
    extract_documents_for_singlestore,
)


def load_json_files(path):
    data_list = []
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(os.path.join(path, filename), "r", encoding="utf-8") as file:
                data = json.load(file)
                data_list.append(data)
    return data_list


def get_bm25_retriever():

    load_dotenv()
    pinecone_params = init_pinecone_index(
        index_name="globalmacro-chatbot",  # Pinecone 인덱스 이름
        namespace="financical-data-00",  # Pinecone Namespace
        api_key=os.environ["PINECONE_API_KEY"],  # Pinecone API Key
        sparse_encoder_path="../data/sparse_encoder_01.pkl",  # Sparse Encoder 저장경로(save_path)
        stopwords=stopwords(),  # 불용어 사전
        tokenizer="kiwi",
        embeddings=UpstageEmbeddings(
            model="solar-embedding-1-large-query"
        ),  # Dense Embedder
        top_k=10,  # Top-K 문서 반환 개수
        alpha=0.4,  # alpha=0.75로 설정한 경우, (0.75: Dense Embedding, 0.25: Sparse Embedding)
    )

    pinecone_retriever = PineconeKiwiHybridRetriever(**pinecone_params)

    return pinecone_retriever


def get_multivector_retriever():
    config = load_yaml("../config/embedding.yaml")
    category_id = config["settings"]["category_id"]
    filetype = config["settings"]["filetype"]
    edit_path = config["settings"]["edit_path"]
    output_path = config["settings"]["output_path"]
    os.makedirs(output_path, exist_ok=True)
    load_dotenv()
    data_list = []
    for category in config["settings"]["category_id"]:
        category_path = os.path.join(edit_path, category, "json")
        data_list.extend(load_json_files(category_path))

    vectorstore_documents = []
    docstore_documents = []
    for data in data_list:
        docstore_document = extract_documents_for_multidocstore(data)
        docstore_documents.extend(docstore_document)
        vectorstore_document = extract_documents_for_multivectorstore(data)
        vectorstore_documents.extend(vectorstore_document)

    passage_embeddings = UpstageEmbeddings(model="solar-embedding-1-large-passage")

    proj_name = f"2024.09.14_for_multivector_store"
    store = LocalFileStore(
        f"/Users/youngseoklee/Desktop/workplace/MacroAgent-withRAG/cache/{proj_name}/data"
    )

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=passage_embeddings,
        document_embedding_cache=store,
        namespace=passage_embeddings.model,
    )
    DB_PATH = f"/Users/youngseoklee/Desktop/workplace/MacroAgent-withRAG/db/2024.09.14_for_multivector_store"
    db = Chroma(persist_directory=DB_PATH, embedding_function=cached_embedder)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    id_key = "id_key"
    doc_ids = [str(uuid.uuid4()) for _ in docstore_documents]

    multi_vector_retriever = MultiVectorRetriever(
        vectorstore=db,
        byte_store=store,
        id_key=id_key,
    )

    summary_docs = [
        Document(
            page_content=s.page_content, metadata={id_key: doc_ids[i], **s.metadata}
        )
        for i, s in enumerate(vectorstore_documents)
    ]
    filtered_summary_docs = filter_complex_metadata(documents=summary_docs)
    filtered_docstore_documents = filter_complex_metadata(documents=docstore_documents)
    multi_vector_retriever.vectorstore.add_documents(
        summary_docs, ids=[doc.metadata[id_key] for doc in summary_docs]
    )
    multi_vector_retriever.docstore.mset(list(zip(doc_ids, docstore_documents)))

    return multi_vector_retriever


def get_parentdocument_retriever():
    config = load_yaml("../config/embedding.yaml")
    category_id = config["settings"]["category_id"]
    filetype = config["settings"]["filetype"]
    edit_path = config["settings"]["edit_path"]
    output_path = config["settings"]["output_path"]
    os.makedirs(output_path, exist_ok=True)
    load_dotenv()
    data_list = []
    for category in config["settings"]["category_id"]:
        category_path = os.path.join(edit_path, category, "json")
        data_list.extend(load_json_files(category_path))

    all_documents = []
    for data in data_list:
        document = extract_documents_for_singlestore(data)
        all_documents.extend(document)

    def _key_encoder(key: int | str) -> str:
        return str(key)

    def _value_serializer(value: float) -> str:
        return pickle.dumps(value)

    def _value_deserializer(serialized_value: str) -> float:
        return pickle.loads(serialized_value)

    proj_name = f"2024.09.14_for_parentdocument_store"
    store = LocalFileStore(
        f"/Users/youngseoklee/Desktop/workplace/MacroAgent-withRAG/cache/{proj_name}/data"
    )

    encoder_store = EncoderBackedStore(
        store=store,
        key_encoder=_key_encoder,
        value_serializer=_value_serializer,
        value_deserializer=_value_deserializer,
    )

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    passage_embeddings = UpstageEmbeddings(model="solar-embedding-1-large-passage")

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=passage_embeddings,
        document_embedding_cache=store,
        namespace=passage_embeddings.model,
    )
    DB_PATH = f"/Users/youngseoklee/Desktop/workplace/MacroAgent-withRAG/db/2024.09.14_for_parentdocument_store"
    db = Chroma(persist_directory=DB_PATH, embedding_function=cached_embedder)

    Parent_retriever = ParentDocumentRetriever(
        vectorstore=db,
        docstore=encoder_store,
        child_splitter=child_splitter,
    )

    filtered_all_documents = filter_complex_metadata(documents=all_documents)

    Parent_retriever.add_documents(
        documents=filtered_all_documents, ids=None, add_to_docstore=True
    )

    return Parent_retriever
