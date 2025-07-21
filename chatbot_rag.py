import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
import os
import uuid

st.set_page_config(page_title="📄 Streaming PDF Chat")

# --- Callback to stream token-by-token ---
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")  # Typing cursor effect

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# --- Sidebar Upload ---
st.sidebar.title("📤 Upload PDF")
upload_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])

# --- PDF Processing ---
def process_pdf(upload):
    temp_folder = "temp"
    os.makedirs(temp_folder, exist_ok=True)
    pdf_path = os.path.join(temp_folder, upload.name)

    with open(pdf_path, "wb") as f:
        f.write(upload.read())

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embedding = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embedding=embedding)
    return vectorstore.as_retriever()

# --- Handle Upload ---
if upload_file and st.session_state.qa_chain is None:
    with st.spinner("Processing PDF..."):
        retriever = process_pdf(upload_file)
        llm = ChatOllama(model="llama3", streaming=True)
        st.session_state.qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# --- Chat UI ---
if st.session_state.qa_chain:
    user_query = st.chat_input("Ask something about the PDF...")

    if user_query:
        st.session_state.chat_history.append(("user", user_query))
        st.chat_message("user").markdown(user_query)

        # Streaming response
        with st.chat_message("assistant"):
            response_container = st.empty()
            stream_handler = StreamHandler(response_container)

            result = st.session_state.qa_chain.run(user_query, callbacks=[stream_handler])
            st.session_state.chat_history.append(("assistant", result))

    # Chat history
    for sender, msg in st.session_state.chat_history[:-1]:  # exclude last since it's already printed
        st.chat_message(sender).markdown(msg)
