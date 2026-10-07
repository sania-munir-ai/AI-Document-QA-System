from loader import load_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_PATH = "documents/AI_RAG_Practice_Document.pdf"


def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    return chunks


if __name__ == "__main__":
    text = load_pdf(PDF_PATH)

    chunks = split_text(text)

    print("📄 PDF loaded!")
    print(f"📦 Total chunks: {len(chunks)}")

    print("\n--- FIRST CHUNK ---\n")
    print(chunks[0])