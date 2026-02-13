import os
import sys

# Make sure backend modules are accessible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import RAGSystem


def main():
    # Path to docs folder (relative to backend)
    docs_path = os.path.join(os.path.dirname(__file__), "..", "docs")
    docs_path = os.path.abspath(docs_path)

    print("====================================")
    print(" HybridRAG System")
    print("====================================")
    print(f"Loading documents from: {docs_path}")
    print("Initializing system... (this may take a minute)\n")

    # Initialize RAG
    rag = RAGSystem(docs_path)

    print("\nSystem ready!")
    print("Type your question (type 'exit' to quit)\n")

    while True:
        query = input("Question: ").strip()

        if query.lower() in ["exit", "quit"]:
            print("Exiting HybridRAG.")
            break

        if query == "":
            continue

        try:
            answer, sources = rag.answer_question(query)

            print("\n===== ANSWER =====")
            print(answer)

            print("\n===== SOURCES =====")
            if sources:
                for s in sources:
                    print("-", s)
            else:
                print("No sources")

            print("\n------------------------------------\n")

        except Exception as e:
            print("Error during processing:", str(e))


if __name__ == "__main__":
    main()
