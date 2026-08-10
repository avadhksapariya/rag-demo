import sys

from dotenv import load_dotenv

# Automatically load environment variables from .env file
load_dotenv()

from src.ingest import process_pdf_and_ingest
from src.query import answer_question
from src.config import PDF_PATH, DB_PATH


def print_menu():
    print("=" * 45)
    print("        RAG DEMO - GEMINI & CHROMADB       ")
    print("=" * 45)
    print("1. Ingest PDF (Process & Embed)")
    print("2. Ask a Question")
    print("3. Exit")


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
            query = input("\n🔍 Enter your question: ").strip()
            if not query:
                print("Question cannot be empty.\n")
                continue

            try:
                print("\nThinking...")
                res = answer_question(query)

                print("\n" + "=" * 45)
                print("🤖 ANSWER:")
                print("=" * 45)
                print(res["answer"])

                print("\n📌 SOURCES RETRIEVED:")
                for idx, doc in enumerate(res["context"]):
                    page = doc.metadata.get("page", "N/A")
                    print(f"  [{idx+1}] Page {page}: {doc.page_content[:120]}...")
                print("\n")
            except Exception as e:
                print(f"❌ Error during query execution: {e}\n")

        elif choice == "3":
            print("Goodbye!")
            sys.exit(0)

        else:
            print("Invalid choice. Please select 1, 2, or 3.\n")


if __name__ == "__main__":
    main()
