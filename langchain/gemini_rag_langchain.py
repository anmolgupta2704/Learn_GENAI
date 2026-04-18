from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA


def gemini_rag():
    """
    RAG system using Google Gemini instead of OpenAI
    """

    # ------------------------------------------
    # STEP 1: DATA
    # ------------------------------------------
    texts = [
        "AI is the simulation of human intelligence",
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks"
    ]

    # ------------------------------------------
    # STEP 2: EMBEDDINGS
    # ------------------------------------------
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # ------------------------------------------
    # STEP 3: VECTOR DB
    # ------------------------------------------
    db = FAISS.from_texts(texts, embeddings)

    retriever = db.as_retriever()

    # ------------------------------------------
    # STEP 4: LLM
    # ------------------------------------------
    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    # ------------------------------------------
    # STEP 5: QA CHAIN
    # ------------------------------------------
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    # ------------------------------------------
    # STEP 6: QUERY
    # ------------------------------------------
    query = "What is AI?"

    result = qa.run(query)

    print("\n🤖 Answer:")
    print(result)


if __name__ == "__main__":
    gemini_rag()