# -*- coding: utf-8 -*-
"""Extract text and lightweight metadata from local research materials."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from docx import Document
from PIL import Image
from pypdf import PdfReader


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def extract_docx(path: Path, max_chars: int) -> dict[str, object]:
    doc = Document(str(path))
    paragraphs = [clean_text(p.text) for p in doc.paragraphs]
    paragraphs = [p for p in paragraphs if p]
    text = "\n".join(paragraphs)
    return {
        "path": str(path),
        "type": "docx",
        "paragraph_count": len(paragraphs),
        "char_count": len(text),
        "preview": text[:max_chars],
    }


def extract_pdf(path: Path, max_chars: int, max_pages: int | None) -> dict[str, object]:
    reader = PdfReader(str(path))
    page_count = len(reader.pages)
    page_limit = page_count if max_pages is None else min(page_count, max_pages)
    chunks = []
    for idx in range(page_limit):
        try:
            chunks.append(reader.pages[idx].extract_text() or "")
        except Exception as exc:
            chunks.append(f"[page {idx + 1} extraction failed: {exc}]")
    text = clean_text("\n".join(chunks))
    meta = reader.metadata or {}
    return {
        "path": str(path),
        "type": "pdf",
        "page_count": page_count,
        "pages_extracted": page_limit,
        "metadata": {key: str(value) for key, value in meta.items()},
        "char_count_extracted": len(text),
        "preview": text[:max_chars],
    }


def extract_image(path: Path) -> dict[str, object]:
    with Image.open(path) as img:
        return {
            "path": str(path),
            "type": "image",
            "format": img.format,
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract research material text previews.")
    parser.add_argument("--paper-dir", default="paper")
    parser.add_argument("--output", default="research_notes/local_paper_extracts.json")
    parser.add_argument("--max-chars", type=int, default=30000)
    parser.add_argument("--max-pages", type=int, default=None)
    args = parser.parse_args()

    paper_dir = Path(args.paper_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for path in sorted(paper_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix == ".docx":
                records.append(extract_docx(path, args.max_chars))
            elif suffix == ".pdf":
                records.append(extract_pdf(path, args.max_chars, args.max_pages))
            elif suffix in {".png", ".jpg", ".jpeg"}:
                records.append(extract_image(path))
        except Exception as exc:
            records.append({"path": str(path), "type": suffix.lstrip("."), "error": str(exc)})

    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = output.with_suffix(".md")
    lines = ["# Local Research Material Extracts", ""]
    for record in records:
        lines.append(f"## {Path(record['path']).name}")
        lines.append("")
        lines.append(f"- Type: {record.get('type')}")
        for key in ["page_count", "pages_extracted", "paragraph_count", "char_count", "char_count_extracted", "width", "height"]:
            if key in record:
                lines.append(f"- {key}: {record[key]}")
        if "error" in record:
            lines.append(f"- error: {record['error']}")
        preview = record.get("preview")
        if preview:
            lines.append("")
            lines.append("```text")
            lines.append(str(preview)[: args.max_chars])
            lines.append("```")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
