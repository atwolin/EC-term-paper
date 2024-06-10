import random
import csv
import numpy as np
import gp
from data import get_embeddings, get_testing_dataset

# seed = 1126
# random.seed(seed)

# python main.py -algo "rf" -e "word2vec" -n 10 -p 250 -c cx_random -pc 1 -pm 0.1 -eval 1000


def planting_rf(dataset, Config, embeddings, cx_method):
    sub_dataset = dataset.sample(frac=0.8, replace=True)

    print(f"Running Random Forest on run {Config.run}")

    one_rf = gp.GP(
        Config.algorithm,
        Config.embedding_type,
        Config.dimension,
        Config.population_size,
        cx_method,
        Config.cross_prob,
        Config.mut_prob,
        Config.num_generations,
        Config.num_evaluations,
        sub_dataset,
        embeddings,
        Config.run,
    )

    one_rf.initialize_pop()
    one_rf.evolving()
    eval_num = one_rf.eval_count

    return one_rf, eval_num


def run_trail(Config):
    data, embeddings, embedding_model = get_embeddings(
        Config.embedding_type, Config.dimension
    )

    cx_method = gp.get_cx_num(Config.crossover_method)

    num_ensemble = 200  # Number of individuals to keep in the ensemble
    num_pick_best = 10  # Number of best individuals to pick from one forest
    ensemble = []  # ensemble to store the best individuals
    total_eval = 0

    for i in range(num_ensemble // num_pick_best):
        one_rf, eval_num = planting_rf(data, Config, embeddings, cx_method)
        top = sorted(one_rf.pop, key=lambda x: x.fitness.values, reverse=True)[:10]
        ensemble.append(top)
        total_eval += eval_num
        if total_eval >= Config.num_evaluations:
            break

    print("Starting testing...")
    gp.ensemble_testing(ensemble, Config, embedding_model)
    return


def random_forest(config):
    for num in range(30):
        config.run = num + 1
        run_trail(config)
    return
