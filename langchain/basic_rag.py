# pip install langchain faiss-cpu openai

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI

# -------------------------
# STEP 1: Create sample data
# -------------------------
texts = [
    "Python is a programming language",
    "Java is used for backend development",
    "Machine Learning is part of AI"
]

# -------------------------
# STEP 2: Convert text → embeddings
# -------------------------
embeddings = OpenAIEmbeddings()

# -------------------------
# STEP 3: Store in vector DB (FAISS)
# -------------------------
db = FAISS.from_texts(texts, embeddings)

# -------------------------
# STEP 4: Search similar text
# -------------------------
query = "What is Python?"
docs = db.similarity_search(query)

# -------------------------
# STEP 5: Use LLM to answer
# -------------------------
llm = ChatOpenAI()

context = docs[0].page_content   # top result

prompt = f"Answer based on this: {context}\nQuestion: {query}"

response = llm.predict(prompt)

print(response)