# Install required libraries (run once)
# pip install langchain openai faiss-cpu tiktoken

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# -------------------------------
# STEP 1: Load Documents
# -------------------------------
# Load a text file (you can replace with PDF, web, etc.)
loader = TextLoader("sample.txt")
documents = loader.load()

# -------------------------------
# STEP 2: Split Documents into Chunks
# -------------------------------
# LLMs work better with smaller chunks
text_splitter = CharacterTextSplitter(
    chunk_size=500,     # max size of each chunk
    chunk_overlap=50    # overlap to preserve context
)

docs = text_splitter.split_documents(documents)

# -------------------------------
# STEP 3: Create Embeddings
# -------------------------------
# Convert text → numerical vectors
embeddings = OpenAIEmbeddings()

# -------------------------------
# STEP 4: Store in Vector Database (FAISS)
# -------------------------------
# FAISS stores vectors for fast similarity search
vector_db = FAISS.from_documents(docs, embeddings)

# -------------------------------
# STEP 5: Create Retriever
# -------------------------------
# Retrieves top-k relevant documents
retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}  # return top 3 relevant chunks
)

# -------------------------------
# STEP 6: Load LLM (Generative AI)
# -------------------------------
llm = ChatOpenAI(
    temperature=0,   # deterministic output
    model="gpt-3.5-turbo"
)

# -------------------------------
# STEP 7: Create Retrieval QA Chain
# -------------------------------
# Combines retriever + LLM
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# -------------------------------
# STEP 8: Ask Question
# -------------------------------
query = "What is the main topic of the document?"

# Get answer using RAG pipeline
response = qa_chain.run(query)

# -------------------------------
# STEP 9: Output Result
# -------------------------------
print("Answer:", response)