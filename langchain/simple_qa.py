import os
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "your-api-key-here"

def simple_qa():
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    while True:
        question = input("\n❓ Ask something (type 'exit' to quit): ")

        if question.lower() == "exit":
            break

        response = llm.invoke(question)
        print("🤖:", response.content)


if __name__ == "__main__":
    simple_qa()