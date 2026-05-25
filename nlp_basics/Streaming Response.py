from langchain_google_genai import ChatGoogleGenerativeAI
import time


def streaming_demo():
    """
    Simulates streaming output like ChatGPT typing
    """

    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    prompt = "Explain AI in simple terms"

    response = llm.invoke(prompt)

    print("\n🤖 Streaming Response:\n")

    # Print word by word (fake streaming)
    for word in response.content.split():
        print(word, end=" ", flush=True)
        time.sleep(0.1)  # delay for effect


if __name__ == "__main__":
    streaming_demo()