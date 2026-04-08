from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')


def generate_ngrams(text, n=2):
    """
    Generate n-grams from text.

    n = 2 → bigrams
    n = 3 → trigrams
    """

    # Tokenize text into words
    tokens = word_tokenize(text)

    # Generate n-grams
    n_grams = list(ngrams(tokens, n))

    return n_grams


if __name__ == "__main__":
    text = "GenAI is transforming the world"

    print("Bigrams:")
    print(generate_ngrams(text, 2))
    print("\nTrigrams:")
    print(generate_ngrams(text, 3))
    print("\n4-grams:")
    print(generate_ngrams(text, 4))
    print("\n5-grams:")
    print(generate_ngrams(text, 5))
    print("\n6-grams:")
    print(generate_ngrams(text, 6))