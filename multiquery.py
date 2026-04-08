from langchain_openai import ChatOpenAI


def multi_query():
    """
    This demonstrates how LLM can generate multiple queries
    to improve retrieval quality.

    Useful in advanced RAG systems.
    """

    llm = ChatOpenAI()

    question = "What is machine learning?"

    # Generate multiple variations of same question
    prompt = f"""
    Generate 3 different ways to ask this question:
    {question}
    """

    response = llm.invoke(prompt)

    print("\n🔍 Query Variations:")
    print(response.content)


if __name__ == "__main__":
    multi_query()