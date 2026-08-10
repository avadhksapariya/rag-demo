import sys

from dotenv import load_dotenv

# Automatically load environment variables from .env file
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage

from src.ingest import process_pdf_and_ingest
from src.query import answer_question
from src.config import PDF_PATH, DB_PATH


def print_menu():
    print("=" * 45)
    print("        RAG DEMO - GEMINI & CHROMADB       ")
    print("=" * 45)
    print("1. Ingest PDF (Process & Embed)")
    print("2. Ask a Question (Chat Session)")
    print("3. Exit")


# Runs a multi-turn chat session with memory.
def chat_loop():
    chat_history = []
    print("\n💬 Chat session started! Type 'exit' or 'back' to return to main menu.\n")

    while True:
        query = input("🔍 Enter your question: ").strip()
        if not query:
            print("Question cannot be empty.\n")
            continue

        if query.lower() in ["exit", "back"]:
            print("Ending chat session.\n")
            break

        try:
            print("\nThinking...")
            res = answer_question(query, chat_history=chat_history)

            print("\n" + "=" * 45)
            print("🤖 ANSWER:")
            print("=" * 45)
            print(res["answer"])

            print("\n📌 SOURCES RETRIEVED:")
            for idx, doc in enumerate(res["context"]):
                page = doc.metadata.get("page", "N/A")
                print(f"  [{idx+1}] Page {page}: {doc.page_content[:120]}...")
            print("\n" + "-" * 45 + "\n")

            chat_history.append(HumanMessage(content=query))
            chat_history.append(AIMessage(content=res["answer"]))

        except Exception as e:
            print(f"❌ Error during query execution: {e}\n")


def main():
    while True:
        print_menu()
        choice = input("Enter choice (1-3): ").strip()

        if choice == "1":
            try:
                process_pdf_and_ingest()
            except Exception as e:
                print(f"❌ Error during ingestion: {e}\n")

        elif choice == "2":
            chat_loop()

        elif choice == "3":
            print("Goodbye!")
            sys.exit(0)

        else:
            print("Invalid choice. Please select 1, 2, or 3.\n")


if __name__ == "__main__":
    main()
