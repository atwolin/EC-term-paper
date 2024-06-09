import random
import gp
from data import get_embeddings

seed = 1126
random.seed(seed)


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

    num_archive = 200  # Number of individuals to keep in the archive
    num_pick_best = 10  # Number of best individuals to pick from one forest
    archive = []  # Archive to store the best individuals
    total_eval = 0 

    for i in range(num_archive // num_pick_best):
        one_rf, eval_num = planting_rf(data, Config, embeddings, cx_method)
        top = sorted(one_rf.pop, key=lambda x: x.fitness.values, reverse=True)[:10]
        archive.append(top)
        total_eval += eval_num
        if total_eval >= Config.num_evaluations:
            break

    return


def random_forest(config):
    for num in range(30):
        config.run = num + 1
        run_trail(config)
    return
