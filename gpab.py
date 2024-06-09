import random
import gp
from data import get_embeddings

seed = 1126
random.seed(seed)


def update_weights(trees):
    print(f"Boosting at iteration {trees.num_gen}")
    errors = []
    for idx, tree in enumerate(trees.pop):
        func = trees.compile(tree, trees.pset)


def boosting(trees, boosting_interval):
    while trees.generation < trees.num_generations:
        trees.select()

        # Update the weights of the instances
        if trees.generation % boosting_interval == 0:
            update_weights(trees)


def run_trail(Config):
    data, embeddings, embedding_model = get_embeddings(
        Config.embedding_type, Config.dimension
    )

    cx_method = gp.get_cx_num(Config.crossover_method)

    # Initialize instance weights
    data["weights"] = 1.0 / len(data)
    data["weights_update"] = 1.0 / len(data)

    boosting_interval = 10  # Boosting interval

    ensemble = []  # Ensemble to store the best individuals

    gpab = gp.GP(
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
    gpab.initialize_pop()

    boosting(gpab, boosting_interval)

    return


def gpab(config):
    for run in range(30):
        config.run = run + 1
        run_trail(config)
    return
