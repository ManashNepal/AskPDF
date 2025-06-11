from MCP import ModelContextProtocol
import streamlit as st

def download_chat_history(context_manager: ModelContextProtocol):
    chat_history = context_manager.get_context().get("chat_history", [])
    if chat_history:
        chat_text = "\n".join([f"{'User' if i % 2 == 0 else 'Bot'}: {msg.content}"
                               for i, msg in enumerate(chat_history)])
        st.download_button("Download Chat History", chat_text, file_name="chat_history.txt")