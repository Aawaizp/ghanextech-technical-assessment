from langchain_community.document_loaders import PyPDFLoader
from pypdf.errors import DependencyError, FileNotDecryptedError, PdfReadError


def load_pdf(pdf_path):
    """
    Load a PDF and return one Document object per page.
    """
    try:
        documents = PyPDFLoader(pdf_path).load()
    except FileNotDecryptedError as exc:
        raise ValueError(
            "This PDF is password-protected. Upload an unlocked copy."
        ) from exc
    except DependencyError as exc:
        raise ValueError(
            "This PDF uses unsupported encryption. Export an unlocked copy."
        ) from exc
    except (PdfReadError, OSError) as exc:
        raise ValueError(
            "This file couldn't be read as a PDF. Export a fresh PDF and try again."
        ) from exc
    if not documents or not any(document.page_content.strip() for document in documents):
        raise ValueError(
            "This PDF has no readable text. It may be scanned or blank; "
            "run OCR before uploading it."
        )
    return documents
