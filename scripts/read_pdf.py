import sys
import fitz

def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_pdf.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    text = read_pdf(pdf_path)
    # output to text file
    out_path = pdf_path + ".txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Text extracted to {out_path}")
