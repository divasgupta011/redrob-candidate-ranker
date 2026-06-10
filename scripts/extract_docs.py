#!/usr/bin/env python3
"""Extract plain text from the .docx files in the hackathon bundle.

The bundle ships the JD, README, signals reference and submission spec as Word
documents. This helper converts them to UTF-8 text so they can be read, diffed,
and quoted in the JD-distillation step -- without adding a python-docx dependency
(a .docx is just a zip containing word/document.xml).

Usage:
    python scripts/extract_docs.py data/raw/challenge --out data/raw/challenge/_extracted
"""
from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path


def docx_to_text(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", "ignore")
    # Paragraph + tab boundaries -> whitespace before stripping tags.
    xml = xml.replace("</w:p>", "\n").replace("</w:tab>", "\t").replace("<w:tab/>", "\t")
    text = re.sub(r"<[^>]+>", "", xml)
    for a, b in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&apos;", "'")]:
        text = text.replace(a, b)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path, help="Directory containing .docx files")
    ap.add_argument("--out", type=Path, default=None, help="Output dir (default: <src>/_extracted)")
    args = ap.parse_args()

    out = args.out or (args.src / "_extracted")
    out.mkdir(parents=True, exist_ok=True)

    docs = sorted(args.src.glob("*.docx"))
    if not docs:
        print(f"No .docx files found in {args.src}")
        return
    for doc in docs:
        target = out / (doc.stem + ".txt")
        target.write_text(docx_to_text(doc), encoding="utf-8")
        print(f"{doc.name:32s} -> {target}  ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
