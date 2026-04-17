# Import Gemini LLM wrapper from LangChain
from langchain_google_genai import ChatGoogleGenerativeAI  

# Import for environment variables
import os  

from dotenv import load_dotenv
import os

load_dotenv()


def gemini_basic():
    """
    This function demonstrates basic LLM usage using Google Gemini.

    Gemini = Google ka latest GenAI model (PaLM ka upgraded version)
    """

    # ------------------------------------------
    # STEP 1: INITIALIZE MODEL
    # ------------------------------------------
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",   # Model name
        temperature=0.7       # Creativity level (0 = factual, 1 = creative)
    )

    # ------------------------------------------
    # STEP 2: INPUT PROMPT
    # ------------------------------------------
    prompt = "Explain Generative AI in simple terms"

    # ------------------------------------------
    # STEP 3: GENERATE RESPONSE
    # ------------------------------------------
    response = llm.invoke(prompt)

    # ------------------------------------------
    # STEP 4: PRINT OUTPUT
    # ------------------------------------------
    print("\n🤖 Gemini Response:")
    print(response.content)


# Entry point
if __name__ == "__main__":
    gemini_basic()