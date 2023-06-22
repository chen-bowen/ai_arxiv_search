import re
from io import BytesIO
from typing import List
import streamlit as st
from langchain.docstore.document import Document
from PyPDF2 import PdfReader

import base64
import tempfile
import pathlib


@st.cache_data()
def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"\n(?=[a-z])", " ", text)

        output.append(text)

    return output


@st.cache_data()
def parse_txt(file: BytesIO) -> str:
    text = file.read().decode("utf-8")
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


def clear_submit():
    st.session_state["submit"] = False


def show_pdf(uploaded_file):
    """
    Displays a PDF file in an iframe using Streamlit.

    Parameters:
        uploaded_file (UploadedFile): The uploaded PDF file.

    Returns:
        None
    """
    # Create a temporary directory to store the uploaded file.
    temp_dir = tempfile.TemporaryDirectory()

    # Write the uploaded file to the temporary directory.
    uploaded_temp_path = pathlib.Path(temp_dir.name) / uploaded_file.name
    with open(uploaded_temp_path, "wb") as output_temporary_file:
        output_temporary_file.write(uploaded_file.getbuffer())

    # Open the temporary file and encode it as base64.
    with open(uploaded_temp_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    # Display the PDF in an iframe.
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="900" height="1000" type="application/pdf">'
    st.markdown(pdf_display, unsafe_allow_html=True)
