from typing import List, Tuple
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

SUMMARIZER_PROMPT = """You are a concise conversation memory compressor.
Your job is to condense older dialogue between a User and an Assistant into a brief factual summary.

Current running summary:
{existing_summary}

New dialogue to fold in:
{dialogue_to_compress}

Instructions:
- Keep track of key topics discussed, entities mentioned, questions asked, and critical facts answered.
- Maintain a strictly factual, concise recap (under 100 words).
- Do NOT output preamble, greetings, or markdown formatting. Output ONLY the updated factual summary paragraph.
"""


def _format_messages_for_prompt(messages: List[BaseMessage]) -> str:
    formatted = []
    for msg in messages:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted)


def compress_chat_history(
    chat_history: List[BaseMessage],
    llm,
    keep_recent_pairs: int = 2,
    existing_summary: str = "",
) -> Tuple[List[BaseMessage], str]:
    max_verbatim_messages = keep_recent_pairs * 2

    # If the history is small enough, no compression needed
    if len(chat_history) <= max_verbatim_messages:
        return chat_history, existing_summary

    # Split into older messages (to compress) and recent messages (to keep verbatim)
    cutoff = len(chat_history) - max_verbatim_messages
    older_messages = chat_history[:cutoff]
    recent_messages = chat_history[cutoff:]

    # Format dialogue to fold into the summary
    dialogue_str = _format_messages_for_prompt(older_messages)

    try:
        prompt = ChatPromptTemplate.from_template(SUMMARIZER_PROMPT)
        chain = prompt | llm
        response = chain.invoke(
            {
                "existing_summary": (
                    existing_summary if existing_summary else "No prior history."
                ),
                "dialogue_to_compress": dialogue_str,
            }
        )
        updated_summary = response.content.strip()
    except Exception as e:
        print(
            f"[HISTORY COMPRESSOR] Warning: Summarization failed, keeping raw recent: {e}"
        )
        updated_summary = existing_summary

    return recent_messages, updated_summary
