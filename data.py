import os
import re
import random
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import fasttext
import fasttext.util

cur_path = os.getcwd()
PATH = re.search(r"(.*EC-term-paper)", cur_path).group(0)
# print(PATH)


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

    return None


def load_model(dim):
    """
    load the trained models
    :param dim: dimension of the word embeddings
    :return: word2vec, glove, fastText models
    """
    # Word2Vec model
    word2vec_model = Word2Vec.load(f"{PATH}/model/word2vec/word2vec.{dim}.model")

    # GloVe model
    def load_glove(file_path):
        glove_vectors = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                glove_vectors[word] = vector
        return glove_vectors

    glove_model = load_glove(f"{PATH}/model/glove/vectors.{dim}.txt")

    # FastText model
<<<<<<< HEAD
    fastText_model = fasttext.load_model(
        f"{PATH}/model/fasttext/fastText.model.{dim}.bin"
    )
=======
    fastText_model = fasttext.load_model(f"{PATH}/model/fasttext/fastText.model.{dim}.bin")
>>>>>>> jing0609

    return word2vec_model, glove_model, fastText_model


def get_training_dataset():
    """
    To get a subset of a larger dataset from a text file
    :return: a sub-dataset
    """
    df = pd.read_csv(f"{PATH}/data/train/training.txt", sep="\t", header=None)
    # Get a sub-dataset
    idx = random.sample(range(0, len(df)), int(len(df) * (0.01)))
    data = df.iloc[idx]

    return data


def get_embeddings(model, dim):
    """
    To get the embeddings of the words in the sub-dataset
    :param model: word embedding model
    :param dim: dimension of the word embeddings
    :param partition: index of the partition
    :return: a DataFrame of the embeddings
    """
    # Load the trained models
    word2vec_model, glove_model, fastText_model = load_model(dim)

    # Get the training dataset
    data = get_training_dataset()

    embeddings = {}
    embedding_model = None
    for line in data[0]:
        for word in line.split():
            if model == "word2vec" and word in word2vec_model.wv:
                embeddings[word] = word2vec_model.wv[word]
                embedding_model = word2vec_model
            elif model == "glove" and word in glove_model:
                embeddings[word] = glove_model[word]
                embedding_model = glove_model
            else:  # model == "fasttext" and word in fastText_model:
                embeddings[word] = fastText_model.get_word_vector(word)
                embedding_model = fastText_model

    return data, embeddings, embedding_model


if __name__ == "__main__":
    # Load models
    # word2vec_model, glove_model, fastText_model = load_model(10)

    # Get sub-dataset
    # data = get_subdataset(1)

    # Get the embeddings
    data, embeddings, model = get_embeddings("fasttext", 10, 1)
    print("data: ", type(data))

    # Test the model
    # test_word = "education"
    # print(f"vector for word2vec: {word2vec_model.wv[test_word]}")
    # if test_word in glove_model:
    #     print(f"vector for glove: {glove_model[test_word]}")
    # print(f"vector for fastText: {fastText_model.get_word_vector(test_word)}")
