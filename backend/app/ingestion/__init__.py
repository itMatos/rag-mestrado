"""Etapa 1 do pipeline: PDF -> páginas de texto limpo -> chunks."""

from app.ingestion.chunker import chunk_document, window_bounds
from app.ingestion.models import Chunk, ExtractedDocument, PageText
from app.ingestion.pdf_extractor import PdfExtractionError, extract_pdf, extract_pdf_bytes

__all__ = [
    "Chunk",
    "ExtractedDocument",
    "PageText",
    "PdfExtractionError",
    "chunk_document",
    "extract_pdf",
    "extract_pdf_bytes",
    "window_bounds",
]
