import fitz
from pathlib import Path

pdfs = [
    "Codigo-Geral-Tributario-Lei-21-14.pdf",
    "Lei-Sociedades-Comerciais-Lei-1-04.pdf"
]

base_path = Path("c:/Projectos/TCC/backend/data/raw_pdfs")

for pdf_name in pdfs:
    path = base_path / pdf_name
    if not path.exists():
        print(f"{pdf_name} not found")
        continue
    
    doc = fitz.open(path)
    print(f"File: {pdf_name}")
    print(f"  Pages: {len(doc)}")
    
    # Check first 5 pages text
    for i in range(min(5, len(doc))):
        text = doc[i].get_text("text").strip()
        print(f"  Page {i+1} text length: {len(text)}")
        if len(text) > 0:
            print(f"  Page {i+1} snippet: {text[:100].replace('\n', ' ')}")
    
    doc.close()
