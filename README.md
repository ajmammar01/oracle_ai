# Oracle AI

An interactive document assistant web app built with Python, Streamlit, LangChain, and OpenAI.

## Features
* **Interactive UI:** Built using Streamlit for chat interaction.
* **Vector Search:** Uses ChromaDB and OpenAI Embeddings (`text-embedding-3-small`) to retrieve relevant context.
* **RAG Pipeline:** Integrates LangChain retrieval chains to answer user queries based on ingested documents.

## How to Run

1. **Install dependencies:**
   `pip install streamlit langchain-openai langchain-chroma langchain-classic dotenv`

2. **Set up environment variables:**
   Create a `.env` file in the root folder with your OpenAI key:
   `OPENAI_API_KEY=your_openai_api_key_here`

3. **Ingest data (first time only):**
   `python ingest.py`

4. **Launch the web application:**
   `streamlit run main.py`
