# Import required modules
from langchain_community.document_loaders import TextLoader  # Load files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Split text
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS  # Vector DB

import os

# Set API key (make sure this is set properly)
os.environ["GOOGLE_API_KEY"] = "your-api-key"


def multi_doc_rag():
    """
    Multi-document RAG system:
    - Load multiple files
    - Combine them
    - Create embeddings
    - Retrieve answers
    """

    # --------------------------------------
    # STEP 1: LOAD MULTIPLE DOCUMENTS
    # --------------------------------------
    loader1 = TextLoader("doc1.txt")  # First file
    loader2 = TextLoader("doc2.txt")  # Second file

    docs1 = loader1.load()  # Load file 1
    docs2 = loader2.load()  # Load file 2

    documents = docs1 + docs2  # Combine both documents

    # --------------------------------------
    # STEP 2: SPLIT TEXT INTO CHUNKS
    # --------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,   # max size per chunk
        chunk_overlap=50  # overlap to retain context
    )

    texts = splitter.split_documents(documents)

    # --------------------------------------
    # STEP 3: CREATE EMBEDDINGS
    # --------------------------------------
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    # --------------------------------------
    # STEP 4: STORE IN VECTOR DB
    # --------------------------------------
    db = FAISS.from_documents(texts, embeddings)

    retriever = db.as_retriever()

    # --------------------------------------
    # STEP 5: LLM
    # --------------------------------------
    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    # --------------------------------------
    # STEP 6: QUERY LOOP
    # --------------------------------------
    while True:
        query = input("\n❓ Ask something (type exit): ")

        if query.lower() == "exit":
            break

        # Retrieve relevant docs
        docs = retriever.get_relevant_documents(query)

        # Take best chunk
        context = docs[0].page_content

        # Prompt
        prompt = f"""
        Answer based on this context:
        {context}

        Question: {query}
        """

        # LLM response
        response = llm.invoke(prompt)

        print("\n🤖 Answer:", response.content)


if __name__ == "__main__":
    multi_doc_rag()