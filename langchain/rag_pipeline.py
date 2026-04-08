import os

# Import LLM and Embeddings from LangChain OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Document loader (for loading text files)
from langchain_community.document_loaders import TextLoader

# Text splitter (for breaking large text into chunks)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector database
from langchain.vectorstores import FAISS


# Set your OpenAI API key (IMPORTANT: use .env in real projects)
os.environ["OPENAI_API_KEY"] = "your-api-key-here"


def rag_pipeline():
    """
    This function implements a complete RAG (Retrieval-Augmented Generation) pipeline.

    Steps:
    1. Load document
    2. Split into chunks
    3. Convert text into embeddings
    4. Store embeddings in vector DB
    5. Perform similarity search
    6. Pass context to LLM and generate answer
    """

    # -------------------------------
    # STEP 1: LOAD DOCUMENT
    # -------------------------------
    # Load text file from local system
    loader = TextLoader("sample.txt")

    # Convert file into LangChain document format
    documents = loader.load()

    # documents is a list of Document objects
    # Each document has .page_content (text) and metadata


    # -------------------------------
    # STEP 2: SPLIT TEXT INTO CHUNKS
    # -------------------------------
    # Large text cannot be processed directly → split into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,      # max characters in one chunk
        chunk_overlap=50     # overlapping helps retain context
    )

    # Split documents into smaller chunks
    texts = splitter.split_documents(documents)

    # Now 'texts' contains multiple small chunks of the original document


    # -------------------------------
    # STEP 3: CREATE EMBEDDINGS
    # -------------------------------
    # Embeddings convert text into numerical vectors
    # These vectors help in semantic search (meaning-based search)
    embeddings = OpenAIEmbeddings()


    # -------------------------------
    # STEP 4: STORE IN VECTOR DATABASE (FAISS)
    # -------------------------------
    # FAISS is used to store and search embeddings efficiently
    db = FAISS.from_documents(texts, embeddings)

    # Now all document chunks are stored as vectors


    # -------------------------------
    # STEP 5: USER QUERY + SIMILARITY SEARCH
    # -------------------------------
    # Example query (user question)
    query = "What is this document about?"

    # Find most relevant chunks based on query
    docs = db.similarity_search(query)

    # docs[0] = most relevant chunk
    # This is the core of RAG (retrieval step)


    # -------------------------------
    # STEP 6: PASS CONTEXT TO LLM
    # -------------------------------
    # Initialize LLM
    llm = ChatOpenAI()

    # Extract content from top result
    context = docs[0].page_content

    # Create prompt with context + question
    prompt = f"""
    Answer the question based only on the context below:

    Context:
    {context}

    Question:
    {query}
    """

    # Generate response from LLM
    response = llm.invoke(prompt)

    # Print final answer
    print("\n🤖 Final Answer:")
    print(response.content)


# Entry point of program
if __name__ == "__main__":
    rag_pipeline()