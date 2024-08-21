import PyPDF2
from Create_chunk import chunk_text_langchain

def read_pdf_to_list(pdf_path):
    pdf_text = []
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            pdf_text.append(page.extract_text())
    return pdf_text

def chunk_pdf_text(pdf_path, chunk_size, overlap):
    pdf_text_list = read_pdf_to_list(pdf_path)
    full_text = ' '.join(pdf_text_list)  # Combine all the pages into one text
    return chunk_text_langchain(full_text, chunk_size, overlap)