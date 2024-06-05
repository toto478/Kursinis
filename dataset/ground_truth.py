import os
import pdfplumber

def pdf_to_text(pdf_file_path):
    base_name = pdf_file_path.rsplit('.', 1)[0]
    text_file_path = f"{base_name}.txt"
    
    text_content = []
    
    with pdfplumber.open(pdf_file_path) as pdf:
        for page in pdf.pages:
            text_content.append(page.extract_text())
    with open(text_file_path, "w", encoding="utf-8") as text_file:
        text_file.write("\n".join(filter(None, text_content)))
    
    print(f"Text extracted and saved to {text_file_path}")

directory_path = "C:\\Users\\Simas\\Desktop\\Ataskaitos"

for filename in os.listdir(directory_path):
    if filename.endswith(".pdf"):
        full_path = os.path.join(directory_path, filename)
        pdf_to_text(full_path)
#full_path = 'Half-Yearly-Report-as-of-30-06-2022-as-per-ASF-Regulation-5-2018.pdf'

