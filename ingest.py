from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load the PDF
reader = PdfReader("test.pdf") # Change this to your PDF name
full_text = ""

# 2. Extract text from the first 5 pages to keep it fast
for i in range(min(5, len(reader.pages))):
    full_text += reader.pages[i].extract_text()

# 3. Setup the "Cutter" (Chunker)
# We want 1000 characters per chunk, with 100 characters of 'overlap' 
# so the AI doesn't lose the flow between pieces.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

# 4. Do the splitting
chunks = text_splitter.split_text(full_text)

# 5. Check the result
print(f"I have created {len(chunks)} chunks.")
print("--- Here is the first chunk ---")
print(chunks[0])
