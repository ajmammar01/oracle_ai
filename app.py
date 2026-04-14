import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv


# 2. Setup Backend (Modern Version)
load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_db = Chroma(persist_directory="data/chroma_db", embedding_function=embeddings)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create the prompt for the AI
system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Context: {context}"
)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Build the Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
rag_chain = create_retrieval_chain(vector_db.as_retriever(), question_answer_chain)

# --- Update the response logic at the bottom of app.py ---
if prompt := st.chat_input("Ask me anything..."):
    # ... (history logic remains the same)
    with st.chat_message("assistant"):
        response = rag_chain.invoke({"input": prompt}) # Change this line
        full_response = response["answer"]            # Change this line
        st.markdown(full_response)
