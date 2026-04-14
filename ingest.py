import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 1. Load environment variables (API Keys)
load_dotenv()

def run_ingestion():
    # 2. Path configuration
    # We use the /data folder to keep our project clean
    pdf_path = "data/test.pdf" 
    db_path = "data/chroma_db"

    print(f"--- Starting ingestion for: {pdf_path} ---")

    # 3. Extract text from PDF
    if not os.path.exists(pdf_path):
        print(f"Error: Could not find {pdf_path}. Make sure it's in the data folder!")
        return

    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text

    # 4. Chunking Logic
    # We use a 1000/100 split to maintain context across chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(full_text)
    print(f"Extracted {len(chunks)} text chunks.")

    # 5. Initialize Embedding Model
    # text-embedding-3-small is the most cost-effective model in 2026
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 6. Create and Persist the Vector Database
    print("Creating vector database in /data/chroma_db...")
    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )

    print("✅ Ingestion complete! Your AI now has memory.")

if __name__ == "__main__":
    run_ingestion()
