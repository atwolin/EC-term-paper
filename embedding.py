import re
import sys
import pandas as pd
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import fasttext
import fasttext.util


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


def partion_data():
    os.makedirs("./data/partition", exist_ok=True)

    count = 0
    sentence = []
    with open(f"./data/tokenized_mnh.txt", "r") as f:
        for lines in f.readlines():
            num_words = len(lines.split())
            # print(num_words, lines)
            # if count < 5:
            if num_words == 6:
                sentence.append(lines)

    # Shuffle the sentences
    random.shuffle(sentence)

    # Write the sentences to a file
    for lines in sentence:
        partition_idx = hash(lines) % 100
        # print("idx: ", partition_index)
        # if count < 5:
        with open(f"./data/partition/partition_{partition_idx}.txt", "a") as f:
            f.write(lines)
            # count += 1
    print("Partitioning done!")
    return None


def train_word2vec_model(df, dim):
    # Tokenize the text
    sentences = df["headline_text"].apply(preprocess_text).tolist()
    # Train the model
    model = Word2Vec(sentences, vector_size=dim, window=5, min_count=1, workers=4)
    # Save the model
    model.save(f"./model/word2vec/word2vec.{dim}.model")
    return model


def train_fasttext_model(df, dim):
    # Train the model
    model = fasttext.train_unsupervised(
        "./data/tokenized_mnh.txt", model="skipgram", minn=1, maxn=1, dim=dim
    )
    model.save_model(f"./model/fasttext/fastText.unsup.model.{dim}.bin")
    return model


if __name__ == "__main__":
    data = load_data()

    # Save tokenized data
    # print("start to save tokenized data.\n")
    # save_tokenized_data(data)
    # print("finish saving tokenized data.\n")

    # Partition the data
    # partion_data()

    # Train Word2Vec models
    dim = sys.argv[1] if len(sys.argv) > 2 else 10
    model = train_word2vec_model(data, dim)
    fastText_model = train_fasttext_model(data, dim)
    print("finish training the models.\n")
