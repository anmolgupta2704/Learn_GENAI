from langchain.llms import OpenAI

llm = OpenAI(temperature=0.7)

response = llm.invoke("Explain recursion in simple terms")

print(response)