from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from app.src.query import answer_question
from app.src.config import get_llm

# Golden Benchmark Cases
EVAL_TEST_CASES = [
    {
        "question": "What is the name of the serpent used as a rope during the churning of the ocean?",
        "ground_truth": "The serpent used as a churning rope was Vasuki.",
        "selected_file": "SamudraManthan-ChurningOfTheOcean.pdf",
    },
    {
        "question": "Which mountain was used as the churning rod during Samudra Manthan?",
        "ground_truth": "Mount Mandara was used as the churning rod.",
        "selected_file": "SamudraManthan-ChurningOfTheOcean.pdf",
    },
    {
        "question": "Who was Sati and who was her father?",
        "ground_truth": "Sati was the daughter of King Daksha and the consort of Lord Shiva.",
        "selected_file": "The-Story-of-Shiva-and-Sati.pdf",
    },
]


# Output Schema for Evaluation
class RAGEvaluationScore(BaseModel):
    faithfulness: float = Field(
        description="Score 0.0 to 1.0: Is the answer strictly derived from the context without hallucinations?"
    )
    faithfulness_reason: str = Field(description="Explanation of faithfulness score")
    answer_relevance: float = Field(
        description="Score 0.0 to 1.0: Does the answer directly address the user question?"
    )
    answer_relevance_reason: str = Field(description="Explanation of relevance score")
    context_recall: float = Field(
        description="Score 0.0 to 1.0: Did the retrieved context contain the information matching ground truth?"
    )
    context_recall_reason: str = Field(description="Explanation of recall score")


def evaluate_rag_run():
    print("🚀 Starting Native RAG Pipeline Evaluation...\n")

    parser = JsonOutputParser(pydantic_object=RAGEvaluationScore)
    judge_llm = get_llm()

    eval_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an impartial, expert AI benchmark judge evaluating a RAG system.\n"
                "Evaluate the given generation across three metrics from 0.0 to 1.0:\n"
                "1. Faithfulness: Is the answer completely backed by the retrieved context?\n"
                "2. Answer Relevance: Does the answer clearly and directly address the user's question?\n"
                "3. Context Recall: Does the retrieved context contain the ground truth facts?\n\n"
                "{format_instructions}",
            ),
            (
                "human",
                "Question: {question}\n\n"
                "Ground Truth: {ground_truth}\n\n"
                "Retrieved Context:\n{context}\n\n"
                "Generated Answer:\n{answer}\n",
            ),
        ]
    )

    eval_chain = eval_prompt | judge_llm | parser

    results = []

    for test in EVAL_TEST_CASES:
        query = test["question"]
        file_scope = test["selected_file"]
        print(f"🔍 Testing: '{query}'")

        # Run live query through our RAG chain
        rag_res = answer_question(
            user_query=query,
            chat_history=[],
            selected_file=file_scope,
        )

        answer = rag_res.get("answer", "")
        retrieved_docs = rag_res.get("context", [])
        context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Judge scores
        judge_res = eval_chain.invoke(
            {
                "question": query,
                "ground_truth": test["ground_truth"],
                "context": context_str,
                "answer": answer,
                "format_instructions": parser.get_format_instructions(),
            }
        )

        print(f">>> : {judge_res}")

        results.append(
            {
                "Question": query,
                "Faithfulness": judge_res["faithfulness"],
                "Relevance": judge_res["answer_relevance"],
                "Recall": judge_res["context_recall"],
            }
        )

    # Print Benchmark Summary Table
    print("\n" + "=" * 65)
    print("📊 BENCHMARK EVALUATION RESULTS")
    print("=" * 65)
    for r in results:
        print(f"Query: {r['Question']}")
        print(f"  • Faithfulness:    {r['Faithfulness'] * 100:.0f}%")
        print(f"  • Answer Relevance: {r['Relevance'] * 100:.0f}%")
        print(f"  • Context Recall:   {r['Recall'] * 100:.0f}%\n")


if __name__ == "__main__":
    evaluate_rag_run()
