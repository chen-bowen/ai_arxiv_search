import streamlit as st
from openai.error import OpenAIError

from components.sidebar import create_sidebar
from utils import parse_pdf, parse_txt, clear_submit, show_pdf
from models.inference import embed_docs, get_answer, search_docs, text_to_docs

st.set_page_config(page_title="KnowledgeGPT", page_icon="📖", layout="wide")
st.header("📖ArXiv Research Assistant")

create_sidebar()


uploaded_file = st.file_uploader(
    "Upload a pdf, or txt file",
    type=["pdf", "txt"],
    help="Scanned documents are not supported yet!",
    on_change=clear_submit,
)

index = None
doc = None
if uploaded_file is not None:
    if uploaded_file.name.endswith(".pdf"):
        doc = parse_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".txt"):
        doc = parse_txt(uploaded_file)
    else:
        raise ValueError("File type not supported!")
    text = text_to_docs(doc)
    try:
        with st.spinner("Indexing document... This may take a while⏳"):
            index = embed_docs(text)
        st.session_state["api_key_configured"] = True
    except OpenAIError as e:
        st.error(e._message)

query = st.text_area("Ask a question about the document", on_change=clear_submit)

button = st.button("Submit")


if button or st.session_state.get("submit"):
    if not st.session_state.get("api_key_configured"):
        st.error("Please configure your OpenAI API key!")
    elif not index:
        st.error("Please upload a document!")
    elif not query:
        st.error("Please enter a question!")
    else:
        st.session_state["submit"] = True
        # Output Columns
        answer_col, pdf_col = st.columns(2)
        sources = search_docs(index, query)

        try:
            answer = get_answer(sources, query)
            with answer_col:
                st.markdown("#### Answer with Sources")
                with st.expander(answer["output_text"].split("SOURCES: ")[0]):
                    st.markdown("#### Sources")
                    for source in sources:
                        st.markdown(source.page_content)
                        st.markdown(source.metadata["source"])
                        st.markdown("---")

            with pdf_col:
                show_pdf(uploaded_file)

        except OpenAIError as e:
            st.error(e._message)
