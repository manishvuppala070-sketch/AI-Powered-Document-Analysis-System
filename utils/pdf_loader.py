import pdfplumber
from langchain_core.documents import Document


def extract_text_from_pdf(pdf_path):
    documents = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"page": i + 1}
                    )
                )

    return documents