import random
import gp
from data import get_embeddings

seed = 1126
random.seed(seed)


def run_trail(Config):
    data, embeddings, embedding_model = get_embeddings(
        Config.embedding_type, Config.dimension
    )

    cx_method = gp.get_cx_num(Config.crossover_method)

    simple_gp = gp.GP(
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
    )
    simple_gp.initialize_pop()
    simple_gp.evolving()

    return


def simple_gp(config):
    run_trail(config)
    return
