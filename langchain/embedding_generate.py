from langchain_huggingface import HuggingFaceEmbeddings

# Initialize model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Single query
query_vector = embeddings.embed_query("What is LangChain?")

# Multiple docs
doc_vectors = embeddings.embed_documents([
    "LangChain is a framework for LLMs.",
    "Embeddings represent text as vectors."
])

print("Query Vector:", query_vector[:5])  # first 5 values
print("Document Vector Length:", len(doc_vectors[0]))