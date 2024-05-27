import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
import re


def load_data():
    # Ensure having necessary NLTK data files
    # nltk.download("punkt")
    # Load dataset
    df = pd.read_csv("./data/abcnews-date-text.csv")
    return df


def preprocess_text(text):
    # Remove non-alphabetic characters and covert to lowercase
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    return word_tokenize(text)


def save_tokenized_data(df):
    df["headline_text"] = df["headline_text"].apply(preprocess_text)
    # df["tokenized"] = df["headline_text"].apply(word_tokenize)
    count = 0
    with open("./data/tokenized_mnh.txt", "w", encoding="utf-8") as f:
        for line in df["headline_text"]:
            if count == 0:
                print(" ".join(line))
                count += 1
            f.write(" ".join(line) + "\n")


def train_word2vec_model(df):
    # Tokenize the text
    sentences = df["headline_text"].apply(preprocess_text).tolist()
    # Train the model
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    # Save the model
    model.save("word2vec_model.model")
    return model


def load_model():
    word2vec_model = Word2Vec.load("word2vec_model.model")

    def load_glove(file_path):
        glove_vectors = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                glove_vectors[word] = vector
        return glove_vectors

    glove_vectors = load_glove(
        "/home/nlplab/atwolin/EC/tp/EC-term-paper/data/glove/vectors.txt"
    )
    return word2vec_model, glove_vectors


if __name__ == "__main__":
    data = load_data()
    # print("start to save tokenized data.\n")
    # save_tokenized_data(data)
    # print("finish saving tokenized data.\n")

    # Train Word2Vec models
    # model = train_word2vec_model(data)

    # Load the trained models
    word2vec_model, glove_vectors = load_model()

    # Test the model
    test_word = "australia"
    print(f"vector for word2vec: {word2vec_model.wv[test_word]}")
    if test_word in glove_vectors:
        print(f"vector for glove: {glove_vectors[test_word]}")
