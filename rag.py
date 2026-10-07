import os

import chromadb
from langchain_ollama import OllamaEmbeddings, OllamaLLM


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "uploaded_document"

EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "llama3.2:3b"

TOP_K = 5


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
# LLM
# ============================================================

llm = OllamaLLM(
    model=LLM_MODEL
)


# ============================================================
# CLEAN SOURCE NAME
# ============================================================

def clean_source_name(source):
    """
    Convert a full/temporary PDF path into only
    the PDF filename.
    """

    if not source:
        return "Unknown document"

    source = str(source)

    # Get only the filename
    filename = os.path.basename(source)

    # If basename somehow fails
    if not filename:
        return "Unknown document"

    return filename


# ============================================================
# SEARCH DOCUMENTS
# ============================================================

def search_documents(question, top_k=TOP_K):
    """
    Search ChromaDB and return relevant document chunks
    with source, page and similarity information.
    """

    if not question or not question.strip():
        return []

    # --------------------------------------------------------
    # GET COLLECTION
    # --------------------------------------------------------

    try:

        collection = client.get_collection(
            name=COLLECTION_NAME
        )

    except Exception as e:

        raise RuntimeError(
            "No document has been processed yet. "
            "Please upload and process a PDF first."
        ) from e


    # --------------------------------------------------------
    # CREATE QUESTION EMBEDDING
    # --------------------------------------------------------

    try:

        query_embedding = embeddings.embed_query(
            question.strip()
        )

    except Exception as e:

        raise RuntimeError(
            "Could not create the question embedding. "
            "Please make sure Ollama is running and "
            f"{EMBEDDING_MODEL} is available."
        ) from e


    # --------------------------------------------------------
    # SEARCH CHROMADB
    # --------------------------------------------------------

    try:

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

    except Exception as e:

        raise RuntimeError(
            f"ChromaDB search failed: {e}"
        ) from e


    # --------------------------------------------------------
    # GET RAW RESULTS
    # --------------------------------------------------------

    raw_documents = results.get(
        "documents",
        [[]]
    )

    raw_metadatas = results.get(
        "metadatas",
        [[]]
    )

    raw_distances = results.get(
        "distances",
        [[]]
    )


    # --------------------------------------------------------
    # CHECK RESULTS
    # --------------------------------------------------------

    if not raw_documents or not raw_documents[0]:
        return []


    documents = raw_documents[0]


    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    if raw_metadatas and raw_metadatas[0]:

        metadatas = raw_metadatas[0]

    else:

        metadatas = [
            {}
            for _ in documents
        ]


    # --------------------------------------------------------
    # DISTANCES
    # --------------------------------------------------------

    if raw_distances and raw_distances[0]:

        distances = raw_distances[0]

    else:

        distances = [
            None
            for _ in documents
        ]


    # --------------------------------------------------------
    # BUILD SOURCE DOCUMENTS
    # --------------------------------------------------------

    source_documents = []


    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        # ----------------------------------------------------
        # CLEAN DOCUMENT
        # ----------------------------------------------------

        if not document:
            continue

        document = document.strip()

        if not document:
            continue


        # ----------------------------------------------------
        # CLEAN METADATA
        # ----------------------------------------------------

        metadata = metadata or {}


        # ----------------------------------------------------
        # SOURCE NAME
        # ----------------------------------------------------

        raw_source = metadata.get(
            "source",
            "Unknown document"
        )

        source_name = clean_source_name(
            raw_source
        )


        # ----------------------------------------------------
        # PAGE NUMBER
        # ----------------------------------------------------

        page = metadata.get(
            "page"
        )


        # ----------------------------------------------------
        # SIMILARITY SCORE
        # ----------------------------------------------------

        similarity_score = None

        if distance is not None:

            try:

                similarity_score = 1 / (
                    1 + float(distance)
                )

                similarity_score = max(
                    0.0,
                    min(
                        1.0,
                        similarity_score
                    )
                )

            except (
                ValueError,
                TypeError
            ):

                similarity_score = None


        # ----------------------------------------------------
        # ADD SOURCE
        # ----------------------------------------------------

        source_documents.append(
            {
                "text": document,
                "source": source_name,
                "page": page,
                "distance": distance,
                "similarity": similarity_score
            }
        )


    return source_documents


# ============================================================
# SEARCH DOCUMENTS WITH SOURCES
# ============================================================

def search_documents_with_sources(
    question,
    top_k=TOP_K
):
    """
    Search documents and return:

    - text
    - source name
    - page number
    - similarity score
    """

    return search_documents(
        question,
        top_k
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    documents
):
    """
    Generate an answer using only the
    retrieved document context.
    """

    if not documents:

        return (
            "I could not find enough information in the "
            "uploaded document to answer this question."
        )


    # ========================================================
    # PREPARE CONTEXT
    # ========================================================

    context_sections = []


    for index, document in enumerate(
        documents,
        start=1
    ):

        if isinstance(
            document,
            dict
        ):

            text = document.get(
                "text",
                ""
            )

            source = document.get(
                "source",
                "Unknown document"
            )

            page = document.get(
                "page"
            )

        else:

            text = str(
                document
            )

            source = "Unknown document"

            page = None


        # ----------------------------------------------------
        # CHECK TEXT
        # ----------------------------------------------------

        if not text or not text.strip():
            continue


        # ----------------------------------------------------
        # PAGE
        # ----------------------------------------------------

        if page is not None:

            try:

                page_number = int(page) + 1

            except (
                ValueError,
                TypeError
            ):

                page_number = page


            source_info = (
                f"Source: {source}, "
                f"Page: {page_number}"
            )

        else:

            source_info = (
                f"Source: {source}"
            )


        # ----------------------------------------------------
        # ADD CONTEXT
        # ----------------------------------------------------

        context_sections.append(
            f"""
[Document Section {index}]
{source_info}

{text}
"""
        )


    # ========================================================
    # JOIN CONTEXT
    # ========================================================

    context = "\n".join(
        context_sections
    )


    if not context.strip():

        return (
            "I could not find enough information in the "
            "uploaded document to answer this question."
        )


    # ========================================================
    # RAG PROMPT
    # ========================================================

    prompt = f"""
You are an AI Document Q&A assistant.

Answer the user's question using ONLY the
provided document context.

DOCUMENT CONTEXT
================

{context}

================

USER QUESTION
================

{question}

================

INSTRUCTIONS

1. Answer using the provided document context.

2. Do not invent information.

3. Do not use outside knowledge when the answer
   is not supported by the document.

4. You may combine information from multiple
   document sections.

5. If the document does not contain enough
   information, say:

"I could not find enough information in the
uploaded document to answer this question."

6. Give a clear and direct answer.

7. Keep the answer concise but informative.

8. For definition questions, explain the concept
   clearly using the document.

9. For "how" questions, explain the process
   using the document.

10. For "why" questions, explain the reason
    using the document.

11. Do not mention internal RAG processes,
    embeddings or ChromaDB unless the user
    specifically asks about them.

12. Do not make assumptions about information
    that is not present in the document.

FINAL ANSWER:
"""


    # ========================================================
    # GENERATE RESPONSE
    # ========================================================

    try:

        answer = llm.invoke(
            prompt
        )

    except Exception as e:

        raise RuntimeError(
            "Could not generate an answer with Ollama. "
            "Please make sure Ollama is running and "
            f"{LLM_MODEL} is available."
        ) from e


    # ========================================================
    # CLEAN ANSWER
    # ========================================================

    if not answer:

        return (
            "I could not generate an answer "
            "from the document."
        )


    answer = str(
        answer
    ).strip()


    if not answer:

        return (
            "I could not generate an answer "
            "from the document."
        )


    return answer