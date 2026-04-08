from transformers import pipeline


def generate_text():
    """
    This function uses a pre-trained LLM for text generation.

    Pipeline:
    - Input prompt
    - Model generates continuation
    """

    # Load text generation pipeline
    generator = pipeline("text-generation", model="gpt2")

    # Prompt
    prompt = "Artificial Intelligence is"

    # Generate text
    result = generator(prompt, max_length=50, num_return_sequences=1)

    print("\n🧠 Generated Text:")
    print(result[0]['generated_text'])
    
    

if __name__ == "__main__":
    generate_text()