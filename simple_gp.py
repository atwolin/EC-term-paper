import random
import gp
from tqdm import tqdm
from data import get_embeddings
import os
import re

cur_path = os.getcwd()
PATH = re.search(r"(.*EC-term-paper)", cur_path).group(0)

seed = 1126
random.seed(seed)


def run_trail(Config):
    data, embeddings, embedding_model = get_embeddings(
        Config.embedding_type, Config.dimension
    )

    cx_method = gp.get_cx_num(Config.crossover_method)

    simple_gp = gp.GP(
        Config.algorithm,
        Config.embedding_type,
        Config.dimension,
        Config.population_size,
        cx_method,
        Config.cross_prob,
        Config.mut_prob,
        Config.num_generations,
        Config.num_evaluations,
        data,
        embeddings,
        Config.run,
    )
    simple_gp.initialize_pop()
    csv_name = simple_gp.evolving()

    ind = simple_gp.hof[0]
    fit = simple_gp.hof[0].fitness.values[0]

    return ind, fit, csv_name


def simple_gp(config):
    best_ind = None
    best_fit = float(0)
    name = None
    for i in tqdm(range(30)):
        config.run = i + 1
        ind, fit, name = run_trail(config)
        if fit > best_fit:
            best_fit = fit
            best_ind = ind

    os.makedirs(f"{PATH}/best_ind_records/{config.embedding_type}/", exist_ok=True)
    with open(f"{PATH}/best_ind_records/{name}_best.txt", "w") as f:
        f.write(f"Best fitness: {best_fit}\n")
        f.write(f"Best individual: {best_ind}\n")
    return
