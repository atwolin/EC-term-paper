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

    print("Starting testing...")
    # Get test data
    test_data, test_embeddings = get_testing_dataset(
        Config.embedding_type, Config.dimension
    )
    tree = gp.GP(
        Config.algorithm,
        Config.embedding_type,
        Config.dimension,
        Config.population_size,
        cx_method,
        Config.cross_prob,
        Config.mut_prob,
        Config.num_generations,
        Config.num_evaluations,
        test_data,
        test_embeddings,
        Config.run,
    )

    print(f"Archive: {archive}")
    file_name = "result.archive." + tree.csv_name()
    with open(f"{Config.algorithm}/result/{file_name}.txt", "w") as f:
        for idx, top in enumerate(archive):
            f.write(f"Forest {idx}\n")
            for tree in top:
                f.write(f"{tree}\n")
            f.write("\n")

    """
    # Get the best individuals
    archive = sorted(archive, key=lambda x: x.fitness.values, reverse=True)[:5]

    # Average the predicted vectors
    y_pred_ensemble = np.zeros(len(archive))
    for idx, tree in enumerate(archive):
        # Get predict vecoters of all sentences
        y_pred = gp.get_predict_vec(tree, tree)
        y_pred_ensemble[idx] = y_pred
    avg_y_pred = np.mean(y_pred_ensemble, axis=0)

    # Get the predicted words of all sentences
    words = []
    for vec in avg_y_pred:
        word = gp.get_predict_word(vec, Config.embedding_type, embedding_model)
        words.append(word)
    X = gp.get_X(tree)
    y = gp.get_y(tree)

    # Save the sentences, predicted words, and record
    csv_name = "result." + tree.csv_name()
    with open(f"{Config.algorithm}/result/{csv_name}", "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "input_word",
                "predict_word",
                "real_word",
                "fitness value[0]",
                "fitness value[1]",
                "fitness value[2]",
                "fitness value[3]",
                "fitness value[4]",
                "avg fiteness value",
                "tree[0]",
                "tree[1]",
                "tree[2]",
                "tree[3]",
                "tree[4]",
            ]
        )
        for i in len(test_data):
            row = [
                X[i],
                words[i],
                y[i],
                archive[0].fitness.values,
                archive[1].fitness.values,
                archive[2].fitness.values,
                archive[3].fitness.values,
                archive[4].fitness.values,
                avg_y_pred[i],
                str(archive[0]),
                str(archive[1]),
                str(archive[2]),
                str(archive[3]),
                str(archive[4]),
            ]
            writer.writerow(row)
    """
    return


def random_forest(config):
    for num in range(30):
        config.run = num + 1
        run_trail(config)
    return
