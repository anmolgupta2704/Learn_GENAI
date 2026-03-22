import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = "your-api-key-here"

def prompt_template_demo():
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Explain {topic} in 3 simple points"
    )

    final_prompt = prompt.format(topic="Machine Learning")

    response = llm.invoke(final_prompt)

    print("\n📌 Prompt Output:")
    print(response.content)


if __name__ == "__main__":
    prompt_template_demo()