# Import ChatOpenAI (LLM) and embeddings model
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  

# Import FAISS vector database
from langchain.vectorstores import FAISS  

# Import RetrievalQA chain (RAG pipeline ready-made)
from langchain.chains import RetrievalQA  


def retrieval_qa():
    """
    This function demonstrates a basic Retrieval-Augmented Generation (RAG) system.
    
    Flow:
    1. Store data
    2. Convert into embeddings
    3. Save in vector DB
    4. Retrieve relevant data
    5. Pass to LLM for answer
    """

    # ------------------------------------------
    # STEP 1: CREATE SAMPLE DATA
    # ------------------------------------------
    # This is our "knowledge base"
    texts = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning is a type of machine learning using neural networks",
        "Natural Language Processing deals with text and language"
    ]

    # ------------------------------------------
    # STEP 2: CREATE EMBEDDINGS
    # ------------------------------------------
    # Convert text into numerical vectors
    embeddings = OpenAIEmbeddings()

    # ------------------------------------------
    # STEP 3: STORE IN VECTOR DATABASE (FAISS)
    # ------------------------------------------
    # Convert texts into embeddings and store them
    db = FAISS.from_texts(texts, embeddings)

    # ------------------------------------------
    # STEP 4: CREATE RETRIEVER
    # ------------------------------------------
    # Retriever helps find relevant data from DB
    retriever = db.as_retriever()

    # ------------------------------------------
    # STEP 5: INITIALIZE LLM
    # ------------------------------------------
    # This model will generate final answer
    llm = ChatOpenAI()

    # ------------------------------------------
    # STEP 6: CREATE RETRIEVAL QA CHAIN
    # ------------------------------------------
    # This connects retriever + LLM
    qa = RetrievalQA.from_chain_type(
        llm=llm,            # LLM for answer generation
        retriever=retriever  # Retriever for fetching relevant context
    )

    # ------------------------------------------
    # STEP 7: USER QUERY
    # ------------------------------------------
    query = "What is machine learning?"

    # ------------------------------------------
    # STEP 8: RUN THE PIPELINE
    # ------------------------------------------
    # Internally:
    # 1. Query → embedding
    # 2. Find similar text
    # 3. Pass context + query to LLM
    result = qa.run(query)

    # ------------------------------------------
    # STEP 9: PRINT RESULT
    # ------------------------------------------
    print("\n🤖 Answer:")
    print(result)


# Entry point of program
if __name__ == "__main__":
    retrieval_qa()