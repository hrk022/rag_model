import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.base import BaseCallbackHandler

# -------------------- ENV SETUP --------------------
load_dotenv()
GROQ_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------- STREAM HANDLER --------------------
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

# -------------------- SESSION STATE --------------------
# Initialize variables so they don't reset on every rerun
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# -------------------- PDF PROCESSING --------------------
def process_pdf(uploaded_file):
    # Save the file temporarily
    os.makedirs("temp", exist_ok=True)
    file_path = os.path.join("temp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and Split
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Recursive splitter is better for preserving context
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Vector store
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------- MAIN UI --------------------
st.set_page_config(page_title="PDF Chatbot", page_icon="🤖")
st.title("🤖 Chat with Your PDF (Groq RAG)")

# -------------------- SIDEBAR --------------------
st.sidebar.title("📄 Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

# Reset the chain if a new file is uploaded
if uploaded_file:
    # This logic ensures that if the file changes, the chain is rebuilt
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if "current_file" not in st.session_state or st.session_state.current_file != file_id:
        with st.spinner("Analyzing document..."):
            retriever = process_pdf(uploaded_file)
            
            llm = ChatGroq(
                temperature=0,
                model_name="llama3-70b-8192",
                streaming=True
            )

            memory = ConversationBufferMemory(
                memory_key="chat_history",
                output_key="answer", # Crucial for ConversationalRetrievalChain
                return_messages=True
            )

            st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=True
            )
            st.session_state.current_file = file_id
            st.session_state.messages = [] # Clear chat for new document
            st.sidebar.success("PDF Processed!")

# -------------------- CHAT DISPLAY --------------------
# Show existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------- USER INPUT --------------------
if user_input := st.chat_input("Ask me something about the PDF..."):
    if not st.session_state.qa_chain:
        st.error("Please upload a PDF first!")
    else:
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Assistant response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            stream_handler = StreamHandler(placeholder)
            
            # Use invoke to get response
            response = st.session_state.qa_chain.invoke(
                {"question": user_input},
                {"callbacks": [stream_handler]}
            )
            
            full_response = response["answer"]
            placeholder.markdown(full_response)
            
            # Add assistant message to state
            st.session_state.messages.append({"role": "assistant", "content": full_response})
