from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms."
)

llm = OpenAI()

chain = LLMChain(llm=llm, prompt=prompt)

response = chain.run("linked list")

print(response)