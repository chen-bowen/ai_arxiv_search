import streamlit as st
from langchain.docstore.document import Document
from typing import List, Dict, Any
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from openai.error import AuthenticationError
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from models.prompts import STUFF_PROMPT


@st.cache_data
def text_to_docs(text: str | List[str]) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            separator="\n",
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata[
                "source"
            ] = f"Page {doc.metadata['page']}- Chunk {doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks


@st.cache_data
def embed_docs(_docs: List[Document]) -> VectorStore:
    """Embeds a list of Documents and returns a FAISS index"""

    if not st.session_state.get("OPENAI_API_KEY"):
        raise AuthenticationError(
            "Enter your OpenAI API key in the sidebar. You can get a key at"
            " https://platform.openai.com/account/api-keys."
        )
    # Embed the chunks
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.get("OPENAI_API_KEY"))  # type: ignore
    return FAISS.from_documents(_docs, embeddings)


@st.cache_data
def search_docs(_index: VectorStore, query: str) -> List[Document]:
    """Searches a FAISS index for similar chunks to the query
    and returns a list of Documents."""

    return _index.similarity_search(query, k=5)


@st.cache_data
def get_answer(_docs: List[Document], query: str) -> Dict[str, Any]:
    """Gets an answer to a question from a list of Documents."""

    # Get the answer
    chain = load_qa_with_sources_chain(
        OpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=st.session_state.get("OPENAI_API_KEY"),
        ),  # type: ignore
        chain_type="stuff",
        prompt=STUFF_PROMPT,
    )

    return chain(
        {"input_documents": _docs, "question": query}, return_only_outputs=True
    )
