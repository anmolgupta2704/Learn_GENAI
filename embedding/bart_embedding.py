from sentence_transformers import SentenceTransformer


def generate_embeddings():
    """
    This function generates embeddings using a pre-trained BERT model.

    Embeddings = numerical representation of text
    Used for:
    - semantic search
    - similarity
    - clustering
    """

    # Load pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Sample sentences
    sentences = [
        "I love machine learning",
        "AI is the future",
        "Deep learning is powerful"
    ]

    # Convert sentences into vectors
    embeddings = model.encode(sentences)

    # Print embeddings
    print("\n📊 Embeddings:")
    print(embeddings)

    # Each sentence → vector of numbers


if __name__ == "__main__":
    generate_embeddings()