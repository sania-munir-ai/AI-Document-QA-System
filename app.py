import os
import streamlit as st

from vectorstore import create_vector_database
from rag import (
    search_documents_with_sources,
    generate_answer
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Document Q&A",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# SESSION STATE
# ============================================================

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #9ca3af;
        margin-bottom: 25px;
    }

    .status-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin: 10px 0 20px 0;
    }

    .status-title {
        font-size: 20px;
        font-weight: 600;
    }

    .status-text {
        font-size: 15px;
        margin-top: 6px;
    }

    .answer-title {
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .sources-title {
        font-size: 19px;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🤖 AI Document Q&A System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload one or more PDFs and ask questions using '
    'Retrieval-Augmented Generation (RAG).'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# HOW THE SYSTEM WORKS
# ============================================================

with st.expander("ℹ️ How this system works"):

    st.write(
        """
        **1. Upload** one or more PDF documents.

        **2. Process** the documents.

        **3. Chunking** divides documents into smaller sections.

        **4. Embeddings** convert text into numerical representations.

        **5. ChromaDB** stores document vectors and metadata.

        **6. Retrieval** finds relevant information.

        **7. Llama 3.2** generates the final answer.

        **8. Sources** show document name, page,
        relevance, and retrieved content.

        **9. Citations** show which document and page
        support the answer.
        """
    )


# ============================================================
# UPLOAD DOCUMENTS
# ============================================================

st.subheader("📄 Upload Documents")

uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# ============================================================
# SHOW SELECTED FILES
# ============================================================

if uploaded_files:

    st.success(
        f"📚 {len(uploaded_files)} PDF(s) selected"
    )

    for file in uploaded_files:

        st.write(
            f"📄 **{file.name}** — "
            f"{file.size / 1024:.1f} KB"
        )


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

if uploaded_files:

    if st.button(
        "⚙️ Process Documents",
        use_container_width=True
    ):

        with st.spinner(
            "📚 Processing your documents..."
        ):

            try:

                # ------------------------------------------------
                # DOCUMENTS FOLDER
                # ------------------------------------------------

                documents_folder = "documents"

                os.makedirs(
                    documents_folder,
                    exist_ok=True
                )


                # ------------------------------------------------
                # SAVE PDF FILES
                # ------------------------------------------------

                pdf_paths = []

                for uploaded_file in uploaded_files:

                    pdf_path = os.path.join(
                        documents_folder,
                        uploaded_file.name
                    )

                    with open(
                        pdf_path,
                        "wb"
                    ) as file:

                        file.write(
                            uploaded_file.getbuffer()
                        )

                    pdf_paths.append(pdf_path)


                # ------------------------------------------------
                # CREATE VECTOR DATABASE
                # ------------------------------------------------

                total_chunks = create_vector_database(
                    pdf_paths
                )


                # ------------------------------------------------
                # SAVE SESSION STATE
                # ------------------------------------------------

                st.session_state.document_processed = True

                st.session_state.processed_files = [
                    file.name
                    for file in uploaded_files
                ]

                st.session_state.messages = []


                # ------------------------------------------------
                # SUCCESS
                # ------------------------------------------------

                st.success(
                    "✅ Documents processed successfully!"
                )

                st.info(
                    f"📦 Total chunks created: "
                    f"{total_chunks}"
                )


            except Exception as e:

                st.error(
                    f"❌ Error processing documents: {e}"
                )


# ============================================================
# DOCUMENT STATUS
# ============================================================

if st.session_state.document_processed:

    document_count = len(
        st.session_state.processed_files
    )

    st.markdown(
        f"""
        <div class="status-card">

            <div class="status-title">
                📊 Document Status
            </div>

            <div class="status-text">
                ✅ {document_count} document(s) processed
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CHAT SECTION
# ============================================================

if st.session_state.document_processed:

    st.subheader("💬 Ask Questions")


    # ========================================================
    # DISPLAY PREVIOUS CHAT
    # ========================================================

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )


    # ========================================================
    # CHAT INPUT
    # ========================================================

    question = st.chat_input(
        "Ask something about your documents..."
    )


    # ========================================================
    # PROCESS QUESTION
    # ========================================================

    if question:

        # ----------------------------------------------------
        # SAVE USER QUESTION
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        # ----------------------------------------------------
        # DISPLAY USER QUESTION
        # ----------------------------------------------------

        with st.chat_message("user"):

            st.write(question)


        try:

            # =================================================
            # RETRIEVAL
            # =================================================

            with st.spinner(
                "🔎 Searching your documents..."
            ):

                source_documents = (
                    search_documents_with_sources(
                        question
                    )
                )


            # =================================================
            # CHECK RETRIEVED SOURCES
            # =================================================

            if not source_documents:

                answer = (
                    "I could not find enough relevant "
                    "information in the uploaded documents "
                    "to answer this question."
                )

            else:

                # =============================================
                # EXTRACT TEXT FOR LLM
                # =============================================

                document_texts = []

                for source in source_documents:

                    if isinstance(
                        source,
                        dict
                    ):

                        text = source.get(
                            "text",
                            ""
                        )

                        if text:

                            document_texts.append(
                                text
                            )

                    elif isinstance(
                        source,
                        str
                    ):

                        if source.strip():

                            document_texts.append(
                                source
                            )


                # =============================================
                # GENERATE ANSWER
                # =============================================

                with st.spinner(
                    "🤖 Generating answer..."
                ):

                    answer = generate_answer(
                        question,
                        document_texts
                    )


            # =================================================
            # CREATE SOURCE INFORMATION
            # =================================================

            citation_lines = []

            seen_sources = set()

            for source in source_documents:

                if not isinstance(
                    source,
                    dict
                ):
                    continue


                # ---------------------------------------------
                # SOURCE NAME
                # ---------------------------------------------

                source_name = source.get(
                    "source",
                    "Unknown document"
                )

                source_name = os.path.basename(
                    source_name
                )


                # ---------------------------------------------
                # PAGE NUMBER
                # ---------------------------------------------

                page = source.get(
                    "page"
                )

                if page is not None:

                    try:

                        page_number = int(page) + 1

                    except (
                        ValueError,
                        TypeError
                    ):

                        page_number = page

                    citation = (
                        f"📄 {source_name} "
                        f"• Page {page_number}"
                    )

                else:

                    citation = (
                        f"📄 {source_name}"
                    )


                # ---------------------------------------------
                # AVOID DUPLICATES
                # ---------------------------------------------

                if citation not in seen_sources:

                    seen_sources.add(
                        citation
                    )

                    citation_lines.append(
                        citation
                    )


            # =================================================
            # SAVE ONLY THE ANSWER IN CHAT HISTORY
            # =================================================

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )


            # =================================================
            # DISPLAY PROFESSIONAL ANSWER
            # =================================================

            with st.chat_message("assistant"):

                st.markdown(
                    '<div class="answer-title">'
                    '🤖 Answer'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.write(answer)


                # =============================================
                # SOURCES USED
                # =============================================

                if citation_lines:

                    st.markdown(
                        '<div class="sources-title">'
                        '📚 Sources Used'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    for citation in citation_lines:

                        st.markdown(
                            f"- {citation}"
                        )


            # =================================================
            # RETRIEVED SOURCES
            # =================================================

            if source_documents:

                with st.expander(
                    "📚 View Retrieved Sources",
                    expanded=False
                ):

                    for i, source in enumerate(
                        source_documents,
                        start=1
                    ):

                        # -------------------------------------
                        # SOURCE CARD
                        # -------------------------------------

                        st.markdown(
                            f"## 📌 Source {i}"
                        )


                        # -------------------------------------
                        # SOURCE NAME
                        # -------------------------------------

                        source_name = source.get(
                            "source",
                            "Unknown document"
                        )

                        source_name = os.path.basename(
                            source_name
                        )


                        # -------------------------------------
                        # PAGE NUMBER
                        # -------------------------------------

                        page = source.get(
                            "page"
                        )

                        if page is not None:

                            try:

                                page_number = int(page) + 1

                            except (
                                ValueError,
                                TypeError
                            ):

                                page_number = page


                            st.caption(
                                f"📄 {source_name} "
                                f"• 📃 Page {page_number}"
                            )

                        else:

                            st.caption(
                                f"📄 {source_name}"
                            )


                        # -------------------------------------
                        # SIMILARITY / RELEVANCE
                        # -------------------------------------

                        similarity = source.get(
                            "similarity"
                        )

                        if similarity is not None:

                            try:

                                similarity_value = float(
                                    similarity
                                )

                                if similarity_value <= 1:

                                    similarity_value *= 100

                                similarity_value = max(
                                    0,
                                    min(
                                        100,
                                        similarity_value
                                    )
                                )

                                st.caption(
                                    f"🎯 Relevance: "
                                    f"{similarity_value:.1f}%"
                                )

                            except (
                                ValueError,
                                TypeError
                            ):

                                pass


                        # -------------------------------------
                        # RETRIEVED CONTENT
                        # -------------------------------------

                        st.markdown(
                            "**Retrieved content:**"
                        )

                        source_text = source.get(
                            "text",
                            ""
                        )

                        if source_text:

                            st.markdown(
                                f"> {source_text}"
                            )

                        else:

                            st.caption(
                                "No retrieved text available."
                            )


                        # -------------------------------------
                        # DIVIDER
                        # -------------------------------------

                        if i < len(
                            source_documents
                        ):

                            st.divider()


        # =====================================================
        # ERROR HANDLING
        # =====================================================

        except RuntimeError as e:

            st.error(
                f"⚠️ {e}"
            )

        except Exception as e:

            st.error(
                "❌ Something went wrong."
            )

            st.caption(
                f"Technical details: {e}"
            )


# ============================================================
# CLEAR CHAT
# ============================================================

if st.session_state.messages:

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# START NEW DOCUMENT
# ============================================================

if st.session_state.document_processed:

    if st.button(
        "📄 Start New Document",
        use_container_width=True
    ):

        st.session_state.document_processed = False

        st.session_state.processed_files = []

        st.session_state.messages = []

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Built with Python • LangChain • Ollama • "
    "Llama 3.2 • nomic-embed-text • "
    "ChromaDB • Streamlit"
)