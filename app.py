import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.prompts.prompt import PromptTemplate
from langchain_groq import ChatGroq 
from streamlit_chat import message
from typing import Protocol, Dict, Any
import os

load_dotenv()

# Model Context Protocol 
class ModelContextProtocol(Protocol):
    def get_context(self) -> Dict[str, Any]:
        ...
    def set_context(self, context: Dict[str, Any]) -> None:
        ...

class StreamlitContextManager(ModelContextProtocol):
    def __init__(self):
        if "model_context" not in st.session_state:
            st.session_state.model_context = {}

    def get_context(self) -> Dict[str, Any]:
        return st.session_state.model_context

    def set_context(self, context: Dict[str, Any]) -> None:
        st.session_state.model_context.update(context)

# Main Functions 
def get_pdf_text(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len
    )
    return splitter.split_text(raw_text)

def get_vector_store(chunks):
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

def get_custom_prompt(user_prompt):
    return PromptTemplate(
        input_variables=["user_prompt", "context", "question"],
        template=f"""
        {user_prompt}

        You must answer questions using **only the provided context**.  
        **Do not use or rely on any previous chat history unless it is clearly relevant.**  
        **Do not generate answers beyond the provided context.**

        Answer requirements:
        - Keep responses **detailed and simple**.
        - If the user asks for simpler wording, use easy language.
        - If the user asks for a brief answer, still provide a complete and helpful explanation.
        - If the answer cannot be found in the context, clearly say so.

        Formatting Guidelines:
        2. Use **bold** for key terms or concepts.
        3. Use **numbered or bulleted lists** when listing points. 
        4. Use **newlines** generously to improve readability.
        5. Use *italics* where needed.
        6. Format **math equations** clearly.
        7. Use **headings** if the answer has multiple sections.

        ---
        Context:
        {{context}}

        Question:
        {{question}}

        Answer:
        """
    )

def get_conversation_chain(vector_store, custom_prompt):
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        verbose=True
    )

def handle_user_input(user_question, context_manager: ModelContextProtocol):
    conversation = context_manager.get_context().get("conversation")
    if conversation:
        response = conversation.invoke(user_question)
        chat_history = response["chat_history"]
        context_manager.set_context({"chat_history": chat_history})

        for i, msg in enumerate(chat_history):
            is_user = i % 2 == 0
            message(msg.content, is_user=is_user, key=f"message{i}")

def download_chat_history(context_manager: ModelContextProtocol):
    chat_history = context_manager.get_context().get("chat_history", [])
    if chat_history:
        chat_text = "\n".join([f"{'User' if i % 2 == 0 else 'Bot'}: {msg.content}"
                               for i, msg in enumerate(chat_history)])
        st.download_button("Download Chat History", chat_text, file_name="chat_history.txt")

# Main App
def main():
    st.set_page_config(page_title="AskPDF", page_icon=":books:")
    context_manager = StreamlitContextManager()

    st.header("Chat with PDFs :books:")

    user_question = st.text_input("Ask a question about your PDF")
    if st.button("Answer", key="answer_button"):
        if context_manager.get_context().get("conversation"):
            with st.spinner("Thinking..."):
                handle_user_input(user_question, context_manager)
        else:
            st.warning("Please upload and process a PDF first!")

    if context_manager.get_context().get("chat_history"):
        download_chat_history(context_manager)

    with st.sidebar:
        st.subheader("Your Documents")
        pdfs = st.file_uploader("Upload your PDFs and click on 'Process'", accept_multiple_files=True)
        user_prompt = st.text_area("How do you want your chatbot to behave?", height=200)

        if st.button("Launch", key="launch_button"):
            with st.spinner("Launching..."):
                raw_text = get_pdf_text(pdfs)
                chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(chunks)
                custom_prompt = get_custom_prompt(user_prompt)
                conversation = get_conversation_chain(vector_store, custom_prompt)

                context_manager.set_context({
                    "conversation": conversation,
                    "chat_history": [],
                    "custom_prompt": custom_prompt,
                    "vector_store": vector_store
                })
                st.success("Conversation launched!")

        st.subheader("Reset Options")
        if st.button("Reset Conversation"):
            context_manager.set_context({
                "conversation": None,
                "chat_history": []
            })
            st.success("Conversation and memory reset!")

if __name__ == "__main__":
    main()
