import os

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "uploaded_document"

EMBEDDING_MODEL = "nomic-embed-text:latest"


# ============================================================
# CHROMADB
# ============================================================

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)


# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

def create_vector_database(pdf_paths):
    """
    Create a ChromaDB collection from one or more PDFs.

    Accepts:
        - one PDF path as a string
        - multiple PDF paths as a list
    """

    # --------------------------------------------------------
    # Convert single path into a list
    # --------------------------------------------------------

    if isinstance(pdf_paths, str):

        pdf_paths = [
            pdf_paths
        ]

    if not pdf_paths:

        raise ValueError(
            "No PDF files were provided."
        )


    # --------------------------------------------------------
    # Load all PDFs
    # --------------------------------------------------------

    all_documents = []

    for pdf_path in pdf_paths:

        if not os.path.exists(pdf_path):

            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )


        # Get the real PDF filename
        original_filename = os.path.basename(
            pdf_path
        )


        loader = PyPDFLoader(
            pdf_path
        )

        documents = loader.load()


        # ----------------------------------------------------
        # Add original filename to metadata
        # ----------------------------------------------------

        for document in documents:

            document.metadata["source"] = (
                original_filename
            )

            all_documents.append(
                document
            )


    # --------------------------------------------------------
    # Check documents
    # --------------------------------------------------------

    if not all_documents:

        raise ValueError(
            "No text could be extracted from the PDF files."
        )


    # --------------------------------------------------------
    # Split documents into chunks
    # --------------------------------------------------------

    chunks = []

    chunk_size = 1000
    chunk_overlap = 150

    for document in all_documents:

        text = document.page_content.strip()

        if not text:

            continue


        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[
                start:end
            ].strip()


            if chunk_text:

                # Make a clean copy of metadata
                metadata = dict(
                    document.metadata
                )


                # Make sure source is only filename
                metadata["source"] = os.path.basename(
                    metadata.get(
                        "source",
                        "Unknown document"
                    )
                )


                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": metadata
                    }
                )


            start += (
                chunk_size
                - chunk_overlap
            )


    # --------------------------------------------------------
    # Check chunks
    # --------------------------------------------------------

    if not chunks:

        raise ValueError(
            "No usable text chunks were created."
        )


    # --------------------------------------------------------
    # Delete old Chroma collection
    # --------------------------------------------------------

    try:

        client.delete_collection(
            name=COLLECTION_NAME
        )

    except Exception:

        pass


    # --------------------------------------------------------
    # Create new collection
    # --------------------------------------------------------

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )


    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]


    metadatas = [
        chunk["metadata"]
        for chunk in chunks
    ]


    ids = [
        f"chunk_{index}"
        for index in range(
            len(texts)
        )
    ]


    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    try:

        vectors = embeddings.embed_documents(
            texts
        )

    except Exception as e:

        raise RuntimeError(
            "Could not create document embeddings. "
            "Make sure Ollama is running and "
            "nomic-embed-text:latest is available."
        ) from e


    # --------------------------------------------------------
    # Store in ChromaDB
    # --------------------------------------------------------

    try:

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=vectors,
            metadatas=metadatas
        )

    except Exception as e:

        raise RuntimeError(
            f"Could not store documents in ChromaDB: {e}"
        ) from e


    # --------------------------------------------------------
    # Return number of chunks
    # --------------------------------------------------------

    return len(texts)