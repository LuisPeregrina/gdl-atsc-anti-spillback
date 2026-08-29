from pathlib import Path

import pdf_inspector

dataset_dir = Path(__file__).resolve().parent.parent / "dataset"

for pdf_path in sorted(dataset_dir.glob("*.pdf")):
	result = pdf_inspector.process_pdf(str(pdf_path)) # type: ignore
	markdown = result.markdown or ""
	markdown_path = pdf_path.with_suffix(".md")
	markdown_path.write_text(markdown, encoding="utf-8")
	print(f"{pdf_path.name}: {result.pdf_type} -> {markdown_path.name}")
