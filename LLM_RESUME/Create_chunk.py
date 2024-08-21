from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text_langchain(text, chunk_size, overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    chunks = splitter.split_text(text)
    return chunks
