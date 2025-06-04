import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.llms import OllamaLLM  
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.prompts.prompt import PromptTemplate

from streamlit_chat import message

def get_pdf_text(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        length_function=len
    )
    chunks = splitter.split_text(raw_text)
    return chunks

def get_vector_store(chunks):
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_store

def get_custom_prompt():
    custom_prompt = PromptTemplate.from_template("""
    You are a helpful assistant that answers questions using **only the provided context**. 
    **Do not use or rely on any previous chat history unless it is clearly relevant.**  
    **Do not generate answers beyond the provided context.**

    Use the question exactly as given. Do not rephrase or reinterpret it.

    Answer requirements:
    - Keep responses **detailed yet simple**.
    - If the user asks for simpler wording, use easy language.
    - If the user asks for a brief answer, still provide a complete and helpful explanation.
    - If the answer cannot be found in the context, clearly say so.

    Formatting Guidelines:
    1. Add a **clear heading** before the answer.
    2. Use **bold** for key terms or concepts.
    3. Use **numbered or bulleted lists** when listing points.
    4. Use **newlines** generously to improve readability.
    5. Use *italics* where needed.
    6. Format **math equations** clearly.
    7. Use **headings** if the answer has multiple sections.

    ---

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    return custom_prompt


def get_conversation_chain(vector_store, custom_prompt):
    llm = OllamaLLM(model="llama3:8b")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt" : custom_prompt},  
        verbose=True
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation.invoke(user_question)
    st.session_state.chat_history = response["chat_history"]

    for i, chat_message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            message(chat_message.content, is_user=True, key= f"message{i}")
        else:
            message(chat_message.content, is_user=False, key= f"message{i}")

def download_chat_history():
    if st.session_state.chat_history:
        chat_text = "\n".join([f"{'User' if i % 2 == 0 else 'Bot'}: {msg.content}"
                               for i, msg in enumerate(st.session_state.chat_history)])
        st.download_button("Download Chat History", chat_text, file_name="chat_history.txt")

def main():
    st.set_page_config(page_title="AskPDF", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None 
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with PDFs :books:")

    user_question = st.text_input("Ask a question about your PDF")
    if st.button(label="Answer", key="answer_button"):
        if st.session_state.conversation:
            with st.spinner("Thinking"):
                handle_user_input(user_question)
        else:
            st.warning("Please Upload PDF first!")

    if st.session_state.chat_history:
        download_chat_history()

    with st.sidebar:
        st.subheader("Your Documents")
        pdfs = st.file_uploader(label="Upload your PDFs and click on 'Process'", accept_multiple_files=True)
        if st.button(label="Process", key="process_button"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdfs)
                chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(chunks)
                custom_prompt = get_custom_prompt()
                st.session_state.conversation = get_conversation_chain(vector_store, custom_prompt)

        st.subheader("Reset Options")
        if st.button("Reset Conversation"):
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.success("Conversation and memory reset!")

if __name__ == "__main__":
    main()
