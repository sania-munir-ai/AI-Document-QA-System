from loader import load_pdf
from chunker import split_text
from langchain_ollama import OllamaEmbeddings


PDF_PATH = "documents/AI_RAG_Practice_Document.pdf"


embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


def create_embeddings(chunks):
    return embeddings.embed_documents(chunks)


if __name__ == "__main__":
    text = load_pdf(PDF_PATH)

    chunks = split_text(text)

    print(f"📦 Total chunks: {len(chunks)}")

    vectors = create_embeddings(chunks)

    print(f"🔢 Total embeddings: {len(vectors)}")
    print(f"📐 Embedding dimensions: {len(vectors[0])}")

    print("\n✅ Embeddings created successfully!")