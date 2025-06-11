from langchain_core.prompts.prompt import PromptTemplate

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