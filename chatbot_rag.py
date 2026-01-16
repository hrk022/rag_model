import os
import streamlit as st
from dotenv import load_dotenv


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI
from langchain_community.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.base import BaseCallbackHandler


  # ✅ Import from your Python file

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  # Now this is defined

os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

# --- Stream Handler for live output ---
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# --- Sidebar: Upload PDF ---
st.sidebar.title("Upload PDF")
upload_file = st.sidebar.file_uploader("Choose a PDF", type="pdf")

# --- Process Uploaded PDF ---
def process_pdf(upload):
    path = os.path.join("temp", upload.name)
    os.makedirs("temp", exist_ok=True)
    with open(path, "wb") as f:
        f.write(upload.read())

    loader = PyPDFLoader(path)
    docs = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    st.sidebar.write("Sample content:", docs[0].page_content[:500])
    return vectorstore.as_retriever()

# --- Title ---
st.title("🤖 Chat with Your PDF (RAG_MODEL)")

# --- Initialize Chain ---
if upload_file and st.session_state.qa_chain is None:
    with st.spinner("Processing PDF and loading model..."):
        retriever = process_pdf(upload_file)

        llm = ChatOpenAI(
            model_name="llama3-70b-8192",
            temperature=0,
            streaming=True,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base="https://api.groq.com/openai/v1"
        )

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=False
        )

        st.session_state.retriever = retriever
        st.session_state.qa_chain = qa_chain

# --- Chat UI ---
if st.session_state.qa_chain:
    user_query = st.chat_input("Ask something about the PDF...")

    if user_query:
        st.session_state.chat_history.append(("user", user_query))
        st.chat_message("user").markdown(user_query)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            stream_handler = StreamHandler(response_placeholder)

            result = st.session_state.qa_chain.invoke(
                {"question": user_query},
                config={"callbacks": [stream_handler]}
            )

            answer = result.get("answer", "") if isinstance(result, dict) else result
            st.session_state.chat_history.append(("assistant", answer))

    for sender, msg in st.session_state.chat_history[:-1]:
        st.chat_message(sender).markdown(msg)




