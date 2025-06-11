import streamlit as st
from dotenv import load_dotenv
from MCP import StreamlitContextManager
from split_embed import get_pdf_text, get_text_chunks, get_vector_store
from custom_prompt import get_custom_prompt
from conversation_chain import get_conversation_chain
from handle_input import handle_user_input
from download_chat import download_chat_history

load_dotenv()

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
