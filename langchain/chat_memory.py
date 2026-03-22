import os
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

os.environ["OPENAI_API_KEY"] = "your-api-key-here"

def chat_with_memory():
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    memory = ConversationBufferMemory()

    conversation = ConversationChain(
        llm=llm,
        memory=memory
    )

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            break

        response = conversation.predict(input=user_input)
        print("Bot:", response)


if __name__ == "__main__":
    chat_with_memory()