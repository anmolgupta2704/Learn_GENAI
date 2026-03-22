import os
from langchain_openai import ChatOpenAI

# Set your API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

def basic_llm():
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    response = llm.invoke("Explain what is Generative AI in simple terms")

    print("\n🤖 LLM Response:")
    print(response.content)


if __name__ == "__main__":
    basic_llm()