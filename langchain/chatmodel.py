from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

chat = ChatOpenAI()

response = chat.invoke([
    HumanMessage(content="What is AI?")
])

print(response.content)