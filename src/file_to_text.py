from pathlib import Path
import pypdf
import docx


def pdf_to_text(file_path: str) -> str:
    reader = pypdf.PdfReader(file_path)
    pages = [page.extract_text() for page in reader.pages if page.extract_text()]
    return "\n".join(pages)


def docx_to_text(file_path: str) -> str:
    doc = docx.Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paragraphs)


def txt_to_text(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8")

EXTRACTORS = {
".pdf": pdf_to_text,
".docx": docx_to_text,
".txt": txt_to_text,
}
    
def file_to_text(file_path: str) -> str:
    """Convert a pdf, docx, or txt file to a string."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = path.suffix.lower()
    extractor = EXTRACTORS.get(ext)
    if extractor is None:
        raise ValueError(f"Unsupported file type: {ext}")
    return extractor(file_path)


def list_supported_files(directory: str) -> list[Path]:
    """Return paths of all supported files in a directory."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    return [f for f in sorted(dir_path.iterdir()) if f.suffix.lower() in EXTRACTORS]


def dir_to_texts(directory: str) -> list[str]:
    """Extract text from all supported files in a directory."""
    return [file_to_text(str(f)) for f in list_supported_files(directory)]
