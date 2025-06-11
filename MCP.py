from typing import Protocol, Dict, Any
import streamlit as st


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