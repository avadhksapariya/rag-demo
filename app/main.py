import sys

from dotenv import load_dotenv

# Automatically load environment variables from .env file
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage

from app.src.ingest import ingest_all_pdfs_in_directory, get_indexed_documents
from app.src.query import answer_question
from app.src.config import PDF_PATH, DB_PATH


def print_menu():
    print("=" * 45)
    print("        RAG DEMO - GEMINI & CHROMADB       ")
    print("=" * 45)
    print("1. Ingest Default PDF")
    print("2. Ask a Question (Chat Session)")
    print("3. List Indexed Documents")
    print("4. Exit")


# Runs a multi-turn chat session with memory and optional document filtering.
def chat_loop():
    chat_history = []

    # Fetch available indexed documents
    indexed_docs = get_indexed_documents()
    selected_file = "All Documents"

    if indexed_docs:
        print("\n📚 Available Documents in Vector DB:")
        print("  [0] All Documents (Query across everything)")
        for idx, doc_name in enumerate(indexed_docs, 1):
            print(f"  [{idx}] {doc_name}")

        filter_choice = input(
            f"\nSelect document filter (0-{len(indexed_docs)}) [Default: 0]: "
        ).strip()
        if filter_choice.isdigit() and 1 <= int(filter_choice) <= len(indexed_docs):
            selected_file = indexed_docs[int(filter_choice) - 1]

    print(f"\n💬 Chat session started! Scope: [{selected_file}]")
    print("Type 'exit' or 'back' to return to main menu.\n")

    while True:
        query = input("🔍 Enter your question: ").strip()
        if not query:
            print("Question cannot be empty.\n")
            continue

        if query.lower() in ["exit", "back"]:
            print("Ending chat session.\n")
            break

        try:
            print("\nThinking...", end="", flush=True)

            # Request streaming generator
            stream_gen = answer_question(
                query,
                chat_history=chat_history,
                selected_file=selected_file,
                stream=True,
            )

            full_answer = []
            retrieved_sources = []
            first_token = True

            for chunk_data in stream_gen:
                if chunk_data["type"] == "token":
                    if first_token:
                        # Erase "Thinking..." and print the header
                        sys.stdout.write("\r" + " " * 20 + "\r")
                        print("=" * 45)
                        print("🤖 ANSWER:")
                        print("=" * 45)
                        first_token = False

                    token = chunk_data["content"]
                    sys.stdout.write(token)
                    sys.stdout.flush()
                    full_answer.append(token)

                elif chunk_data["type"] == "sources":
                    retrieved_sources = chunk_data["content"]

            print("\n")

            # Display source citations
            if retrieved_sources:
                print("📌 SOURCES RETRIEVED:")
                for idx, doc in enumerate(retrieved_sources):
                    page = doc.metadata.get("page", 0) + 1
                    source_file = doc.metadata.get("source_file", "Document")
                    snippet = doc.page_content.strip().replace("\n", " ")[:120]
                    print(f"  [{idx+1}] {source_file} - Page {page}: {snippet}...")
                print("\n" + "-" * 45 + "\n")

            # Update conversation memory
            final_text = "".join(full_answer)
            chat_history.append(HumanMessage(content=query))
            chat_history.append(AIMessage(content=final_text))

        except Exception as e:
            print(f"\n❌ Error during query execution: {e}\n")


def main():
    while True:
        print_menu()
        choice = input("Enter choice (1-4): ").strip()

        if choice == "1":
            try:
                ingest_all_pdfs_in_directory()
            except Exception as e:
                print(f"❌ Error during ingestion: {e}\n")

        elif choice == "2":
            chat_loop()

        elif choice == "3":
            docs = get_indexed_documents()
            print("\n📚 Indexed Documents:")
            if docs:
                for doc in docs:
                    print(f"  • {doc}")
            else:
                print("  No documents indexed yet.")
                print("\n")

        elif choice == "4":
            print("Goodbye!")
            sys.exit(0)

        else:
            print("Invalid choice. Please select 1, 2, 3, or 4.\n")


if __name__ == "__main__":
    main()
