# 🧠 AI PDF Chatbot using Llama 3 via Groq & LangChain

🚀 Talk to your PDFs with blazing speed using **Meta's Llama 3 (70B)** served via **Groq API**, integrated with **LangChain** and deployed on **Streamlit Cloud**.

[![Streamlit App](https://img.shields.io/badge/Launch%20App-Click%20Here-00bfff?style=for-the-badge&logo=streamlit)](https://ragmodel-numqugkseyjfkbg9thfn7o.streamlit.app/)

---

## 📌 Features

- ✅ Upload any PDF and ask questions instantly
- ⚡ Powered by **Groq’s ultra-fast inference**
- 🧠 Uses **Llama 3 70B** for high-quality answers
- 🔗 Built with **LangChain** & **FAISS vector search**
- ☁️ Fully deployed and accessible via Streamlit

---

## 🛠️ Tech Stack

| Layer        | Technology           |
|--------------|----------------------|
| Model        | 🦙 LLaMA 3 (70B)      |
| Inference    | 🚀 Groq API           |
| Framework    | 🧱 LangChain          |
| Embedding    | 🧬 Sentence Transformers |
| Vector Store | 📚 FAISS              |
| PDF Parsing  | 📄 PyPDF              |
| Frontend     | 🌐 Streamlit          |
| Deployment   | 🛸 Streamlit Cloud    |

---

## 🧪 Try it Now

📎 Upload your PDF and start chatting:  
👉 [**Open the Live App**](https://ragmodel-numqugkseyjfkbg9thfn7o.streamlit.app/)

---

## 🔐 Environment Variables

Secrets are managed securely using Streamlit's secret manager.

Set them in `.streamlit/secrets.toml` (in Streamlit dashboard, not in repo):

```toml
OPENAI_API_KEY = "your_groq_api_key"
OPENAI_API_BASE = "https://api.groq.com/openai/v1"
