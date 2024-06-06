import re
import os
import random
import pandas as pd
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import fasttext
import fasttext.util

cur_path = os.getcwd()
PATH = re.search(r"(.*EC-term-paper)", cur_path).group(0)
print(PATH)


def load_data():
    # Ensure having necessary NLTK data files
    # nltk.download("punkt")
    # Load dataset
    df = pd.read_csv(f"{PATH}/data/abcnews-date-text.csv")
    return df


def preprocess_text(text):
    # Remove non-alphabetic characters and covert to lowercase
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    return word_tokenize(text)


def save_tokenized_data(df):
    df["headline_text"] = df["headline_text"].apply(preprocess_text)
    count = 0
    with open(
        f"{PATH}/data/tokenized_mnh.txt",
        "w",
        encoding="utf-8",
    ) as f:
        for line in df["headline_text"]:
            if count == 0:
                print(" ".join(line))
                count += 1
            f.write(" ".join(line) + "\n")


def process_six_words_data():
    sentence = []
    with open(f"{PATH}/data/tokenized_mnh.txt", "r") as f:
        for lines in f.readlines():
            if len(lines.split()) == 6:
                sentence.append(lines)

    # Shuffle the sentences
    random.shuffle(sentence)

    # Save the dataset of six words
    with open(f"{PATH}/data/six_words.txt", "w") as f:
        for lines in sentence:
            f.write(lines)

    # Save the testing data
    os.makedirs(f"{PATH}/data/test", exist_ok=True)
    with open(f"{PATH}/data/test/testing.txt", "w") as f:
        for lines in sentence[:10000]:
            f.write(lines)

    # Save the traing data
    os.makedirs(f"{PATH}/data/train", exist_ok=True)
    with open(f"{PATH}/data/train/training.txt", "w") as f:
        for lines in sentence[10000:]:
            f.write(lines)

    # Partition the training data
    # os.makedirs(f"{PATH}/data/train", exist_ok=True)
    # for lines in sentence[10000:]:
    #     partition_idx = hash(lines) % 100
    #     with open(f"{PATH}/data/train/partition_{partition_idx}.txt", "a") as f:
    #         f.write(lines)
    # print("Partitioning done!")

    return None


def train_word2vec_model(df, dim):
    # Tokenize the text
    sentences = df["headline_text"].apply(preprocess_text).tolist()
    # Train the model
    model = Word2Vec(sentences, vector_size=dim, window=5, min_count=1, workers=4)
    # Save the model
    model.save(f"{PATH}/model/word2vec/word2vec.{dim}.model")
    print(f"save model at {PATH}/model/word2vec/word2vec.{dim}.model")
    return model


def train_fasttext_model(df, dim):
    # Train the model
    model = fasttext.train_unsupervised(
        f"{PATH}/data/tokenized_mnh.txt", model="skipgram", minn=1, maxn=1, dim=dim
    )
    model.save_model(f"{PATH}/model/fasttext/fastText.model.{dim}.bin")
    print(f"save model at {PATH}/model/fasttext/fastText.model.{dim}.bin")
    return model


if __name__ == "__main__":
    data = load_data()

    # Save tokenized data
    # print("start to save tokenized data.\n")
    # save_tokenized_data(data)
    # print("finish saving tokenized data.\n")

    # Extract sentences of six words and partition the data
    process_six_words_data()

    # Train Word2Vec models
    # dim = int(sys.argv[1] if len(sys.argv) > 1 else 10)
    # print(f"dim is {dim}")
    # word2vec_model = train_word2vec_model(data, dim)
    # fastText_model = train_fasttext_model(data, dim)
    # print("Finish training the models.\n")

    # Test the model
    # test_word = "australia"
    # print(f"vector for word2vec: {word2vec_model.wv[test_word]}")
    # if test_word in glove_vectors:
    #     print(f"vector for glove: {glove_vectors[test_word]}")
    # print(f"vector for fastText: {fastText_model.get_word_vector(test_word)}")
