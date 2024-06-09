import sys
import warnings
import random
import csv
import numpy as np
from deap import gp as deap_gp
import gp
from data import get_embeddings, get_testing_dataset

# seed = 1126
# random.seed(seed)

# python main.py -algo "gpab" -e "word2vec" -n 10 -p 250 -c cx_random -pc 1 -pm 0.1 -g 100


def update_weights(
    gpab,
    best_ind,
    X,
    y,
    iboost,
    sample_weight,
    learning_rate,
    loss,
    ensemble,
    num_ensemble,
):
    y_pred = gp.get_predict_vec(gpab, best_ind)

    error_vect = np.linalg.norm(y - y_pred, axis=1)
    sample_mask = sample_weight > 0
    masked_sample_weight = sample_weight[sample_mask]
    masked_error_vector = error_vect[sample_mask]

    error_max = masked_error_vector.max()
    if error_max != 0:
        masked_error_vector /= error_max

    if loss == "square":
        masked_error_vector **= 2
    elif loss == "exponential":
        masked_error_vector = 1.0 - np.exp(-masked_error_vector)

    # Culcalate the average loss
    estimator_error = (masked_sample_weight * masked_error_vector).sum()
    if estimator_error <= 0:
        # Stop if fit is perfect
        return sample_weight, 1.0, 0.0
    elif estimator_error >= 0.5:
        # Discard the estimator if worse than random guessing and it isn't the only one
        if len(ensemble) > 0:
            ensemble.pop(-1)
        return None, None, None

    beta = estimator_error / (1.0 - estimator_error)

    # Boost weight using AdaBoost.R2 algorithm
    estimator_weight = learning_rate * np.log(1.0 / beta)

    if not iboost == num_ensemble - 1:
        sample_weight[sample_mask] *= np.power(
            beta, (1.0 - masked_error_vector) * learning_rate
        )

    return sample_weight, estimator_weight, estimator_error


def boosting(
    gpab, data, ensemble, num_ensemble, iboost, sample_weight, loss, learning_rate
):
    epsilon = np.finfo(sample_weight.dtype).eps
    zero_weight_mask = sample_weight == 0.0

    # Get the best individual
    best_ind = max(gpab.pop, key=lambda x: x.fitness.values)
    # Avoid extremely small weights
    sample_weight = np.clip(sample_weight, a_min=epsilon, a_max=None)
    sample_weight[zero_weight_mask] = 0.0

    # Boosting step
    X = gp.get_X(gpab)
    y = gp.get_y(gpab)
    sample_weight, estimator_weight, estimator_error = update_weights(
        gpab,
        best_ind,
        X,
        y,
        iboost,
        sample_weight,
        learning_rate,
        loss,
        ensemble,
        num_ensemble,
    )

    # Early termination
    if sample_weight is None:
        return sample_weight, best_ind

    # Stop if error is zero
    if estimator_error == 0:
        return sample_weight, best_ind

    sample_weight_sum = np.sum(sample_weight)

    if not np.isfinite(sample_weight_sum):
        warnings.warn(
            (
                "Sample weights have reached infinite values,"
                f" at iteration {iboost}, causing overflow. "
                "Iterations stopped. Try lowering the learning rate."
            ),
            stacklevel=2,
        )
        return sample_weight, best_ind

    # Stop if the sum of sample weights has become non-positive
    if sample_weight_sum <= 0:
        return sample_weight, best_ind

    if iboost < num_ensemble - 1:
        # Normalize
        sample_weight /= sample_weight_sum

    print("Sample weight: ", sample_weight)
    data["weights_update"] *= sample_weight

    return sample_weight, best_ind


def select_new_data(gpab, best, data):
    # Update the population dataset
    select_new_data = np.random.uniform(0, 1, len(data))
    data["cumulative_weights"] = data["weights_update"].cumsum()

    # Find the indices of the closest rows in cumulative_weights for each value in select_new_data
    indices = np.digitize(select_new_data, data["cumulative_weights"])

    # Ensure indices are within bounds
    indices = np.clip(indices, 0, len(data) - 1)

    # Create new dataset by selecting rows from original dataset based on indices
    new_dataset = data.iloc[indices].reset_index(drop=True)

    gpab.data = new_dataset
    # Evaluate the entire population
    fitnesses = map(gpab.toolbox.evaluate, gpab.pop)
    for ind, fit in zip(gpab.pop, fitnesses):
        ind.fitness.values = fit


def evolving(
    gpab, data, ensemble, num_ensemble, iboost, sample_weight, loss, learning_rate
):
    while gpab.n_gen < gpab.max_gen:
        gpab.select()

        # Boosting
        if gpab.n_gen % iboost == 0:
            sample_weight, best = boosting(
                gpab,
                data,
                ensemble,
                num_ensemble,
                iboost,
                sample_weight,
                loss,
                learning_rate,
            )
            ensemble.append(best)

            # Update the data
            select_new_data(gpab, best, data)

        gpab.n_gen += 1


def run_trail(Config):
    data, embeddings, embedding_model = get_embeddings(
        Config.embedding_type, Config.dimension
    )

    cx_method = gp.get_cx_num(Config.crossover_method)

    # Initialize instance weights
    data["weights"] = 1.0 / len(data)
    data["weights_update"] = 1.0 / len(data)

    ensemble = []  # Ensemble to store the best individuals
    iboost = 1  # Config.num_generations / 10  # Boosting interval
    num_ensemble = 5
    loss = "linear"
    learning_rate = 1.0
    sample_weight = np.array(data["weights_update"])

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
    evolving(
        gpab, data, ensemble, num_ensemble, iboost, sample_weight, loss, learning_rate
    )

    print("Starting testing...")
    print(f"Ensemble: {ensemble}")
    # Get test data
    test_data, test_embeddings = get_testing_dataset(
        Config.embedding_type, Config.dimension
    )
    gpab.data = test_data
    gpab.embeddings = test_embeddings

    print(f"Archive: {ensemble}")
    file_name = "result.archive." + gpab.csv_name()
    with open(f"{Config.algorithm}/result/{file_name}.txt", "w") as f:
        for idx, top in enumerate(ensemble):
            f.write(f"Forest {idx}\n")
            for tree in top:
                f.write(f"{tree}\n")
            f.write("\n")

    """

    # Get the best individuals
    archive = sorted(ensemble, key=lambda x: x.fitness.values, reverse=True)[:5]
    print("Archive: ", archive)

    # Average the predicted vectors
    y_pred_ensemble = np.zeros(len(archive))
    for idx, tree in enumerate(archive):
        # Get predict vecoters of all sentences
        y_pred = gp.get_predict_vec(gpab, tree)
        y_pred_ensemble[idx] = y_pred
    avg_y_pred = np.mean(y_pred_ensemble, axis=0)

    # Get the predicted words of all sentences
    words = []
    for vec in avg_y_pred:
        word = gp.get_predict_word(vec, Config.embedding_type, embedding_model)
        words.append(word)
    X = gp.get_X(gpab)
    y = gp.get_y(gpab)

    # Save the sentences, predicted words, and record
    csv_name = "result." + gpab.csv_name()
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


def gpab(config):
    for i in range(30):
        config.run = i + 1
        run_trail(config)
    return
