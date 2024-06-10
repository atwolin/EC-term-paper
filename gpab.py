import sys
import os
import re
import warnings
from tqdm import tqdm
import random
import csv
import numpy as np
from deap import gp as deap_gp
import gp
from data import get_embeddings, get_testing_dataset

# seed = 1126
# random.seed(seed)

# python main.py -algo "gpab" -e "word2vec" -n 10 -p 250 -c cx_random -pc 1 -pm 0.1 -g 100

cur_path = os.getcwd()
PATH = re.search(r"(.*EC-term-paper)", cur_path).group(0)
# print(PATH)


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
        return sample_weight, []

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

    # print("Sample weight: ", sample_weight)
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
            if sample_weight is None:
                sample_weight = np.array(data["weights_update"])
                continue
            ensemble.append(best)

            # Update the data
            select_new_data(gpab, best, data)

        # Record the statistics
        csv_name = gpab.csv_name()
        # os.makedirs(f"{PATH}/results", exist_ok=True)
        # print(f"csv_name: {csv_name},run:{self.run}")

        with open(f"{PATH}/results/{csv_name}.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["eval_count", "avg", "std", "min", "max", "best_individual"]
            )
            if gpab.eval_count % 1 == 0:
                gpab.write_record(writer)
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
    print("Starting evolving...")
    evolving(
        gpab, data, ensemble, num_ensemble, iboost, sample_weight, loss, learning_rate
    )

    # for ind in ensemble:
    #     print(f"Individual: {ind}")
    #     print(type(ind))

    print("Starting testing...")
    gp.ensemble_testing(ensemble, Config, embedding_model)
    return


def gpab(config):
    for i in tqdm(range(10)):
        config.run = i + 1
        run_trail(config)
    return
