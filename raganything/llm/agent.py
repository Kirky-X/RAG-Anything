# Copyright (c) 2025 Kirky.X
# All rights reserved.

from typing import Any, List


def build_simple_agent(chat_model: Any, tools: List[Any] | None = None) -> Any:
    """Build a simple agent using the provided chat model and tools.

    Args:
        chat_model (Any): The chat model to use for the agent.
        tools (List[Any] | None): Optional list of tools to bind to the model.

    Returns:
        Any: A LangChain chain representing the simple agent.
    """
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
    except Exception as e:
        raise ValueError(f"LangChain core unavailable: {e}")

    if tools:
        try:
            chat_model = chat_model.bind_tools(tools)
        except Exception:
            pass

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful professional agent."),
            ("human", "{input}"),
        ]
    )

    chain = prompt | chat_model | StrOutputParser()
    return chain
