from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

EXPANSION_PROMPT = """You are an expert search retrieval query optimizer.
Your task is to take a user query and decompose or expand it into 2 to 3 distinct, specific search queries.

Objectives:
- If the question is multi-faceted or comparative (e.g., comparing two entities, events, or concepts), break it down into focused sub-queries for each entity or concept.
- If the question is simple, provide alternative keyword perspectives or synonym queries.
- Keep queries concise (3 to 8 words each).
- Do NOT answer the question.
- Do NOT output preamble, numbering, bullet marks, or commentary.
- Separate each query with a NEWLINE.

Primary User Question:
{question}
"""


def generate_multi_queries(primary_query: str, llm) -> List[str]:
    prompt = ChatPromptTemplate.from_template(EXPANSION_PROMPT)
    chain = prompt | llm | StrOutputParser()

    try:
        response = chain.invoke({"question": primary_query})
        expanded = [q.strip() for q in response.strip().split("\n") if q.strip()]

        all_queries = [primary_query]
        for q in expanded:
            if q.lower() != primary_query.lower() and q not in all_queries:
                all_queries.append(q)
        return all_queries[:3]

    except Exception as e:
        print(
            f"[QUERY EXPANSION] Warning: Expansion failed, defaulting to primary query: {e}"
        )
        return [primary_query]


def deduplicate_documents(documents: List[Document]) -> List[Document]:
    seen_identifiers = set()
    unique_docs = []

    for doc in documents:
        doc_id = doc.metadata.get("doc_id") or doc.metadata.get("source_id")
        if not doc_id:
            source = doc.metadata.get("source_file", "unknown")
            page = doc.metadata.get("page", 0)
            content_snippet = doc.page_content[:100]
            doc_id = f"{source}_{page}_{content_snippet}"

        if doc_id not in seen_identifiers:
            seen_identifiers.add(doc_id)
            unique_docs.append(doc)

    return unique_docs
