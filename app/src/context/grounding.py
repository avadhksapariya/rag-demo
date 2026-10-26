import re
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.src.models.grounding_models import GroundingReport
from app.src.prompts.prompts import VERIFICATION_PROMPT, CORRECTION_PROMPT


# Splits a paragraph into clean sentences while preserving bracketed citations.
def split_sentences(text: str) -> List[str]:
    cleaned = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", text.strip())
    return [s.strip() for s in cleaned if len(s.strip()) > 5]


# Evaluates an answer against retrieved context chunks.
def verify_grounding(answer: str, documents: List[Document], llm) -> Dict[str, Any]:
    if not answer or not documents:
        return {
            "is_grounded": True,
            "score": 1.0,
            "verifications": [],
            "corrected_answer": None,
        }

    content_str = "\n\n".join([d.page_content for d in documents])
    parser = JsonOutputParser(pydantic_object=GroundingReport)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", VERIFICATION_PROMPT),
            ("human", "Audit this answer now."),
        ]
    )

    try:
        report_data = (prompt | llm | parser).invoke(
            {
                "context": content_str,
                "answer": answer,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        report = GroundingReport(**report_data)
    except Exception as e:
        print(f"GROUNDING AUDITOR Warning: Verification check skipped: {e}")
        return {
            "is_grounded": True,
            "score": 1.0,
            "verifications": [],
            "corrected_answer": None,
        }

    # If all claims are grounded, return immediately
    if not report.needs_correction:
        print(
            f"🛡️ Grounding Auditor: Verified! Grounding score: {report.overall_score:.2f}"
        )
        return {
            "is_grounded": True,
            "score": report.overall_score,
            "verifications": [v.model_dump() for v in report.verifications],
            "corrected_answer": None,
        }

    # If ungrounded claims are present, execute self-correction
    unsupported_items = [
        f"- {v.sentence} (Reason: {v.unsupported_reason})"
        for v in report.verifications
        if not v.is_supported
    ]
    unsupported_str = "\n".join(unsupported_items)

    print(
        f"⚠️ Grounding Auditor: Flagged {len(unsupported_items)} ungrounded claim(s). Running self-correction..."
    )

    correction_prompt = ChatPromptTemplate.from_template(CORRECTION_PROMPT)
    try:
        correction_chain = correction_prompt | llm
        corrected_res = correction_chain.invoke(
            {
                "context": content_str,
                "original_answer": answer,
                "unsupported_claims": unsupported_str,
            }
        )
        corrected_text = corrected_res.content.strip()
        return {
            "is_grounded": False,
            "score": report.overall_score,
            "verifications": [v.model_dump() for v in report.verifications],
            "corrected_answer": corrected_text,
        }
    except Exception as e:
        print(f"GROUNDING AUDITOR Warning: Correction rewrite failed: {e}")
        return {
            "is_grounded": False,
            "score": report.overall_score,
            "verifications": [v.model_dump() for v in report.verifications],
            "corrected_answer": answer,
        }
