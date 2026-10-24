from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class UserIntent(BaseModel):
    intent: str = Field(
        description="Classify as 'chitchat' if the message is a greeting, reaction, compliment, pleasantry, acknowledgment (e.g., 'Amazing', 'Thanks', 'Cool', 'Haha wow', 'Understood'), or casual banter. Classify as 'rag_query' if the user is asking a factual question, seeking information, or following up on document content."
    )
    direct_response: str = Field(
        default="",
        description="If intent is 'chitchat', provide a short, warm, natural conversational reply (e.g. 'Glad you found it interesting! Let me know if you need anything else.'). If 'rag_query', leave this empty.",
    )


class DocumentFilterDecision(BaseModel):
    target_file: str | None = Field(
        default=None,
        description="The exact filename from the candidate list that the user specifically mentions or refers to. If the user does not specify a particular file, return None.",
    )


# Classifies user input dynamically using the LLM before any vector retrieval.
def classify_user_intent(user_query: str, llm) -> UserIntent:
    parser = JsonOutputParser(pydantic_object=UserIntent)
    intent_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an input intent classifier for an intelligent assistant.\n"
                "Determine whether the user is casually interacting/acknowledging or asking a genuine question.\n"
                "{format_instructions}",
            ),
            ("human", "{input}"),
        ]
    )
    chain = intent_prompt | llm | parser
    try:
        res = chain.invoke(
            {
                "input": user_query,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        return UserIntent(**res)
    except Exception:
        # Default to standard RAG pipeline if parsing fails
        return UserIntent(intent="rag_query", direct_response="")


# Detects if the user query references a specific document name or theme
# and maps it to an indexed file in ChromaDB.
def detect_document_filter(query: str, available_files: list[str], llm) -> str | None:
    if not available_files:
        return None

    if len(available_files) == 1:
        return available_files[0]

    parser = JsonOutputParser(pydantic_object=DocumentFilterDecision)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a document routing assistant.\n"
                "Analyze the user question and determine if they are asking about or referencing a specific document from this list:\n"
                "{file_list}\n\n"
                "RULES:\n"
                "1. If the user mentions a document title, topic, or file name matching an item in the list, set 'target_file' to that exact file name.\n"
                "2. If the user does not specify or asks across all knowledge, set 'target_file' to null.\n"
                "{format_instructions}",
            ),
            ("human", "{question}"),
        ]
    )

    chain = prompt | llm | parser
    try:
        res = chain.invoke(
            {
                "question": query,
                "file_list": "\n".join([f"- {f}" for f in available_files]),
                "format_instructions": parser.get_format_instructions(),
            }
        )
        return res.get("target_file")
    except Exception:
        return None


# Rephrases follow-up questions into standalone search queries using conversation context.
def reformulate_standalone_query(user_query: str, effective_history: list, llm) -> str:
    if not effective_history:
        return user_query

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a search query optimizer. Given a conversation history and a follow-up user message, "
                "rephrase it into a concise, standalone keyword search query (3 to 8 words maximum).\n"
                "CRITICAL RULES:\n"
                "- DO NOT answer the question.\n"
                "- DO NOT explain or add commentary.\n"
                "- DO NOT use markdown bolding (**), quotes, or punctuation.\n"
                "- Output ONLY the search query string.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    try:
        res = (prompt | llm).invoke(
            {"input": user_query, "chat_history": effective_history}
        )
        return res.content.strip()
    except Exception:
        return user_query
