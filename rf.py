import random
import gp
from data import get_embeddings

seed = 1126
random.seed(seed)


def planting_rf(dataset, Config, embeddings, cx_method):
    sub_dataset = dataset.sample(frac=0.8, replace=True)

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

    return one_rf


def run_trail(Config):
    data, embeddings, embedding_model = get_embeddings(
        Config.embedding_type, Config.dimension
    )

    cx_method = gp.get_cx_num(Config.crossover_method)

    num_archive = 200  # Number of individuals to keep in the archive
    num_pick_best = 10  # Number of best individuals to pick from one forest
    archive = []  # Archive to store the best individuals

    for i in range(num_archive // num_pick_best):
        one_rf = planting_rf(data, Config, embeddings, cx_method)
        top = sorted(one_rf.pop, key=lambda x: x.fitness.values, reverse=True)[:10]
        archive.append(top)

    return


def random_forest(config):
    for run in range(30):
        config.run = run + 1
        run_trail(config)
    return
