from typing import List, Tuple
from langchain_core.documents import Document

CHAR_PER_TOKEN = 4


# Fast, dependency-free token estimator.
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // CHAR_PER_TOKEN)


def trim_documents_to_budget(
    documents: List[Document], max_tokens: int = 2500
) -> Tuple[List[Document], int]:
    selected_docs = []
    accumulated_tokens = 0

    for doc in documents:
        doc_tokens = estimate_tokens(doc.page_content)

        if accumulated_tokens + doc_tokens > max_tokens and selected_docs:
            print(
                f"✂️ [Token Budget]: Exceeded {max_tokens} tokens. Trimmed remaining {len(documents) - len(selected_docs)} chunk(s)."
            )
            break

        selected_docs.append(doc)
        accumulated_tokens += doc_tokens

    return selected_docs, accumulated_tokens
