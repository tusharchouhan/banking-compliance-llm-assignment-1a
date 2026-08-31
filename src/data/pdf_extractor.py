"""Extract every page of every source PDF into a separate UTF-8 text file."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from src.config.settings import SETTINGS
from src.utils.common import configure_logging


def extract_pdf(pdf_path: Path) -> tuple[str, int, int]:
    """Return page-delimited text, page count, and extracted character count."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except ImportError:
        import fitz  # PyMuPDF fallback
        doc = fitz.open(pdf_path)
        pages = [page.get_text("text") or "" for page in doc]
        doc.close()
    text = "\n\n".join(f"[Page {i}]\n{page.strip()}" for i, page in enumerate(pages, 1))
    return text.strip() + "\n", len(pages), len(text)


def run(input_dir: Path, output_dir: Path, report_path: Path, logger: logging.Logger) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # Search recursively so the corpus may be organized by regulator, year,
    # document type, or any other folder taxonomy.
    pdfs = sorted(input_dir.rglob("*.pdf"))
    rows: list[dict[str, object]] = []
    if not pdfs:
        logger.warning("No PDFs found in %s; place the 10-50 MB banking corpus there.", input_dir)
    for pdf in pdfs:
        try:
            text, pages, chars = extract_pdf(pdf)
            relative_pdf = pdf.relative_to(input_dir)
            out = output_dir / relative_pdf.with_suffix(".txt")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            rows.append({"source_pdf": relative_pdf.as_posix(), "output_txt": out.relative_to(output_dir).as_posix(), "pages": pages,
                         "characters": chars, "status": "ok"})
            logger.info("Extracted %s: %d pages, %d characters", pdf.name, pages, chars)
        except Exception as exc:  # keep processing the rest of the corpus
            logger.exception("Failed to extract %s", pdf)
            rows.append({"source_pdf": pdf.name, "output_txt": "", "pages": 0,
                         "characters": 0, "status": f"error: {exc}"})
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_pdf", "output_txt", "pages", "characters", "status"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Extraction complete: %d PDFs", len(pdfs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=SETTINGS.raw_pdfs)
    parser.add_argument("--output-dir", type=Path, default=SETTINGS.extracted_txt)
    parser.add_argument("--report", type=Path, default=SETTINGS.reports_dir / "tables" / "extraction_report.csv")
    args = parser.parse_args()
    run(args.input_dir, args.output_dir, args.report, configure_logging("pdf_extractor"))
