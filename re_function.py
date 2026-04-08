import re


def clean_text(text):
    """
    This function cleans raw text data.

    Steps:
    - Lowercase conversion
    - Remove punctuation
    - Remove numbers
    - Remove extra spaces
    """

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


if __name__ == "__main__":
    sample = "Hello!!! This is GENAI 2025 🚀"

    cleaned = clean_text(sample)

    print("Original:", sample)
    print("Cleaned:", cleaned)