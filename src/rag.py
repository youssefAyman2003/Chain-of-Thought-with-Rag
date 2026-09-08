import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.llm import embeddings

os.environ.setdefault(
    "USER_AGENT",
    "Chain-of-Thought-RAG/0.1.0 (https://github.com/your-org/chain-of-thought-with-rag)",
)

doc=WebBaseLoader("https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/pages/syllabus/").load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50,separators=["\n\n", "\n", " ", ""])
docs = splitter.split_documents(doc)


vectorstore = FAISS.from_documents(docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

