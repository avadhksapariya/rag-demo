import re
from typing import List
from rank_bm25 import BM25Okapi


# Splits a document text into clean sentences.
def _split_into_sentence(text: str) -> List[str]:
    # Split on sentence terminals followed by space and capital letter or bracket
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", text.strip())
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 15]
    return sentences if sentences else [text.strip()]


def extract_highlighted_snippet(
    doc_text: str,
    query: str,
    max_sentences: int = 2,
) -> str:
    if not doc_text:
        return ""

    sentences = _split_into_sentence(doc_text)
    if len(sentences) <= 1:
        # If single sentence or very short, highlight the first meaningful segment
        return f"<mark>{doc_text[:200]}</mark>..."

    tokenized_corpus = [re.findall(r"\w+", s.lower()) for s in sentences]
    tokenized_query = re.findall(r"\w+", query.lower())

    if not tokenized_query:
        return doc_text[:250].replace("\n", " ") + "..."

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_indices = sorted(ranked_indices[:max_sentences])

    highlighted_parts = []
    for idx, sent in enumerate(sentences):
        if idx in top_indices and scores[idx] > 0:
            highlighted_parts.append(
                f"<span style='text-decoration: underline; text-decoration-color: #b2bec3; "
                f"text-decoration-thickness: 1px; text-underline-offset: 3px;'>{sent}</span>"
            )
        else:
            highlighted_parts.append(sent)

    joined = " ".join(highlighted_parts)
    return joined.replace("\n", " ")
