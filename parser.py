import argparse


class Config:
    def __init__(
        self,
        algorithm,
        embedding_type,
        dimension,
        population_size,
        crossover_method,
        cross_prob,
        mut_prob,
        num_generations,
        num_evaluations,
        debug,
        run,
    ):
        self.algorithm = algorithm
        self.embedding_type = embedding_type
        self.dimension = dimension
        self.population_size = population_size
        self.crossover_method = crossover_method
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob
        self.num_generations = num_generations
        self.num_evaluations = num_evaluations
        self.debug = debug
        self.run = run

    def print_configuration(self):
        parameters = {
            "algorithm": self.algorithm,
            "embedding_type": self.embedding_type,
            "dimension": self.dimension,
            "population_size": self.population_size,
            "crossover_method": self.crossover_method,
            "cross_prob": self.cross_prob,
            "mut_prob": self.mut_prob,
            "num_generations": self.num_generations,
            "num_evaluations": self.num_evaluations,
            "num_runs": self.run,
        }

        print("-" * 44)
        print("|{:<20}|{:<20}|".format("Parameter", "Value"))
        print("-" * 44)
        for key, value in parameters.items():
            print("|{:<20}|{:<20}|".format(key, str(value)))
        print("-" * 44)


def str_to_bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="GP Configuration Parser")
    parser.add_argument(
        "-algo",
        "--algorithm",
        choices=["simple_gp", "rf", "gpab"],
        default="simple_gp",
        help="The algorithm to use. simple_gp, rf, or gpab (default: simple_gp)",
    )
    parser.add_argument(
        "-e",
        "--embedding_type",
        choices=["word2vec", "glove", "fasttext"],
        default="word2vec",
        help="The type of embedding model to use. word2vec, glove, or fasttext (default: Word2Vec)",
    )
    parser.add_argument(
        "-n",
        "--dimension",
        type=int,
        default=10,
        help="The dimension of word embeddings (default: 10)",
    )
    parser.add_argument(
        "-p",
        "--population_size",
        type=int,
        default=100,
        help="Number of the population (default: 100)",
    )
    parser.add_argument(
        "-c",
        "--crossover_method",
        choices=[
            "cx_random",  # 0
            "cx_simple",  # 1
            "cx_uniform",  # 2
            "cx_fair",  # 3
            "cx_one_point",  # 4
        ],
        default="cx_random",
        help="The crossover method to use. cx_random, cx_simple, cx_uniform, cx_fair, or cx_one_point(default: cx_random)",
    )
    parser.add_argument(
        "-pc",
        "--prob_crossover",
        type=float,
        default=0.9,
        help="Probability for the crossover (default: 0.9)",
    )
    parser.add_argument(
        "-pm",
        "--prob_mutation",
        type=float,
        default=0.1,
        help="Probability for the mutation (default: 0.1)",
    )
    parser.add_argument(
        "-g",
        "--generations",
        type=int,
        default=500,
        help="Max number of generations to terminate (default: 500)",
    )
    parser.add_argument(
        "-eval",
        "--evaluations",
        type=int,
        default=1000,
        help="Max number of evaluations to terminate (default: 1000)",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Turn on debug prints (default: false)",
    )
    args = parser.parse_args()

    return Config(
        algorithm=args.algorithm,
        embedding_type=args.embedding_type,
        dimension=args.dimension,
        population_size=args.population_size,
        crossover_method=args.crossover_method,
        cross_prob=args.prob_crossover,
        mut_prob=args.prob_mutation,
        num_generations=args.generations,
        num_evaluations=args.evaluations,
        run=1,
        debug=args.debug,
    )
