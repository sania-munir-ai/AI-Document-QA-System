from pypdf import PdfReader


PDF_PATH = "documents/AI_RAG_Practice_Document.pdf"


def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


if __name__ == "__main__":
    text = load_pdf(PDF_PATH)

    print("📄 PDF loaded successfully!")
    print(f"Total characters: {len(text)}")

    print("\n--- PDF TEXT PREVIEW ---\n")
    print(text[:2000])