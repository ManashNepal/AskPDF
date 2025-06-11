from MCP import ModelContextProtocol
from streamlit_chat import message

def handle_user_input(user_question, context_manager: ModelContextProtocol):
    conversation = context_manager.get_context().get("conversation")
    if conversation:
        response = conversation.invoke(user_question)
        chat_history = response["chat_history"]
        context_manager.set_context({"chat_history": chat_history})

        for i, msg in enumerate(chat_history):
            is_user = i % 2 == 0
            message(msg.content, is_user=is_user, key=f"message{i}")