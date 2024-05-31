import sys
import os
import re
import random
import numpy as np
import pandas as pd
import nltk
from gensim.models import Word2Vec
import fasttext
import fasttext.util

cur_path = os.getcwd()
PATH = re.search(r"(.*EC-term-paper)", cur_path).group(0)
print(PATH)


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
    # fastText_model = fasttext.load_model("./model/fasttext/fastText_model.bin")
    fastText_model = fasttext.load_model(
        f"{PATH}/model/fasttext/fastText.model.{dim}.bin"
    )

    return word2vec_model, glove_model, fastText_model


def get_subdataset(partition_idx=1, n=500):
    """
    To get a subset of a larger dataset from a text file
    :param partition_idx: index of the partition
    :param n: number of samples to be selected
    :return: a sub-dataset
    """
    # Load dataset
    df = pd.read_csv(
        f"{PATH}/data/partition/partition_{partition_idx}.txt", sep="\t", header=None
    )
    # Get a sub-dataset
    idx = random.sample(range(0, len(df)), n)
    data = df.iloc[idx]

    return data


def get_embeddings(model, dim, partition=1):
    """
    To get the embeddings of the words in the sub-dataset
    :param model: word embedding model
    :param dim: dimension of the word embeddings
    :param partition: index of the partition
    :return: a DataFrame of the embeddings
    """
    # Load the trained models
    word2vec_model, glove_model, fastText_model = load_model(dim)

    # Get the sub-dataset
    data = get_subdataset(partition)

    count = 0
    embeddings = {}
    for line in data[0]:
        for word in line.split():
            if count < 5:
                # print(word)
                print(f"word: {word}, vector: {word2vec_model.wv[word]}")
                count += 1
            if model == "word2vec" and word in word2vec_model.wv:
                embeddings[word] = word2vec_model.wv[word]
                print(f"word: {word}, vector: {word2vec_model.wv[word]}")
            elif model == "glove" and word in glove_model:
                embeddings[word] = glove_model[word]
            elif model == "fasttext" and word in fastText_model:
                embeddings[word] = fastText_model.get_word_vector(word)
            else:
                embeddings[word] = (
                    word2vec_model.wv[word],
                    glove_model[word],
                    fastText_model.get_word_vector(word),
                )

    df = pd.DataFrame(embeddings)

    return df


if __name__ == "__main__":
    # Get sub-dataset
    data = get_subdataset(1)

    # Get the embeddings
    df = get_embeddings("word2vec", 10, 1)
    print(len(df))

    print("aa")

    # Test the model
    # test_word = "australia"
    # print(f"vector for word2vec: {word2vec_model.wv[test_word]}")
    # if test_word in glove_vectors:
    #     print(f"vector for glove: {glove_vectors[test_word]}")
    # print(f"vector for fastText: {fastText_model.get_word_vector(test_word)}")
