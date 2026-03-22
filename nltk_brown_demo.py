import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize


def download_nltk_resources():
    """
    Download required NLTK datasets
    """
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('brown')


def explore_brown_corpus():
    """
    Explore Brown Corpus categories and sample data
    """
    print("\n Brown Corpus Categories:")
    print(brown.categories())

    print("\n Sample Words from 'news' category:")
    print(brown.words(categories='news')[:10])


def tokenize_sample_text():
    """
    Tokenize a small sample text from Brown Corpus
    """
    text = brown.raw(categories='news')[:200]

    print("\nSample Text:")
    print(text)

    print("\n Tokenized Words:")
    tokens = word_tokenize(text)
    print(tokens)


def main():
    print(" Starting NLTK Brown Corpus Demo...\n")

    download_nltk_resources()
    explore_brown_corpus()
    tokenize_sample_text()

    print("\n NLTK Processing Completed!")


if __name__ == "__main__":
    main()