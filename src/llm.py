from langchain_ollama import ChatOllama, OllamaEmbeddings

llm = ChatOllama(model="llama3.2:3b", temperature=0.2)
embeddings = OllamaEmbeddings(model="bge-m3:latest")
