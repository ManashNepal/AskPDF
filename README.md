# 📚 AskPDF – Chat with Your PDFs

AskPDF is an interactive chatbot app that lets you **ask questions about your PDF documents** and get intelligent, contextual answers. Powered by **LangChain**, **Groq's LLaMA 3.3 70B Versatile model**, and **Streamlit**, this tool transforms static documents into dynamic conversations.

---

## ✨ Features

- 🧠 **Chat with any PDF**: Upload textbooks, manuals, research papers, or any PDF.
- ⚙️ **LLM-powered responses**: Uses Groq’s `llama-3.3-70b-versatile` model for accurate, conversational answers.
- 🔎 **Semantic search**: Retrieves the most relevant parts of your document using FAISS vector store and embeddings.
- 💬 **Memory support**: Maintains conversation context for multi-turn questions.
- 📅 **Downloadable chat history**: Save your conversation as a `.txt` file.
- ⟲ **Session memory reset**: Easily reset the chat and upload new documents.

---

## 🚀 How It Works

1. **PDF Upload**  
   Upload one or more PDFs via the sidebar.

2. **Text Extraction & Chunking**  
   Extracts the full text from the PDFs and splits it into manageable chunks using LangChain’s `RecursiveCharacterTextSplitter`.

3. **Embedding & Indexing**  
   Chunks are converted into vector embeddings using `mxbai-embed-large` and stored in a FAISS index.

4. **Conversational Retrieval Chain**  
   Uses `ConversationalRetrievalChain` with Groq's `llama-3.3-70b-versatile` model to answer questions based on relevant document chunks.

5. **Interactive Chat**  
   Type your question, and the chatbot retrieves the most relevant content and responds intelligently.

---

## 🧠 Powered By

- [Streamlit](https://streamlit.io/) – for the web interface  
- [LangChain](https://www.langchain.com/) – to build the chatbot logic  
- [Groq API](https://console.groq.com/) – for fast and efficient LLM inference  
- [FAISS](https://github.com/facebookresearch/faiss) – for efficient vector search  
- [PyPDF2](https://pypi.org/project/PyPDF2/) – for PDF text extraction  
