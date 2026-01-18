import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.base import BaseCallbackHandler

# -------------------- ENV SETUP --------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = "https://api.groq.com/openai/v1"

os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE

# -------------------- STREAM HANDLER --------------------
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

# -------------------- SESSION STATE --------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# -------------------- SIDEBAR --------------------
st.sidebar.title("📄 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF", type="pdf")

# -------------------- PDF PROCESSING --------------------
def process_pdf(uploaded_file):
    os.makedirs("temp", exist_ok=True)
    file_path = os.path.join("temp", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    st.sidebar.markdown("**Preview:**")
    st.sidebar.write(documents[0].page_content[:500])

    return vectorstore.as_retriever()

# -------------------- MAIN UI --------------------
st.title("🤖 Chat with Your PDF (RAG)")

# -------------------- INIT CHAIN --------------------
if uploaded_file and st.session_state.qa_chain is None:
    with st.spinner("Processing PDF & initializing model..."):
        retriever = process_pdf(uploaded_file)

        llm = ChatOpenAI(
            model_name="llama3-70b-8192",
            temperature=0,
            streaming=True,
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_API_BASE,
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )

        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=False,
        )

# -------------------- CHAT --------------------
if st.session_state.qa_chain:
    user_input = st.chat_input("Ask a question about the PDF...")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        st.chat_message("user").markdown(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            stream_handler = StreamHandler(placeholder)

            result = st.session_state.qa_chain.invoke(
                {"question": user_input},
                config={"callbacks": [stream_handler]},
            )

            answer = result.get("answer", "")
            st.session_state.chat_history.append(("assistant", answer))

    for role, message in st.session_state.chat_history[:-1]:
        st.chat_message(role).markdown(message)
