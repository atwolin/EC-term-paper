import argparse


class Config:
    def __init__(
        self,
        embeddings,
        dimension,
        population_size,
        crossover_method,
        cross_prob,
        mut_prob,
        num_generations,
        debug,
    ):
        self.embeddings = embeddings
        self.dimension = dimension
        self.population_size = population_size
        self.crossover_method = crossover_method
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob
        self.num_generations = num_generations
        # self.crossover_method = self.determine_crossover_method()
        self.debug = debug

    # def determine_crossover_method(self):
    #     if not self.uniform_crossover:
    #         if self.representation == "binary":
    #             return "2-point"
    #         elif self.representation == "real":
    #             return "whole arithmetic"
    #     return "uniform"

    def print_configuration(self):
        parameters = {
            "embedding model": self.embeddings,
            "dimension": self.dimension,
            "population_size": self.population_size,
            "crossover_method": self.crossover_method,
            "cross_prob": self.cross_prob,
            "mut_prob": self.mut_prob,
            "num_generations": self.num_generations,
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
    parser = argparse.ArgumentParser(description="GA Configuration Parser")
    parser.add_argument(
        "-e",
        "--embeddings",
        choices=["word2vec", "glove", "fasttext"],
        default="word2vec",
        help="The embedding model to use. word2vec, glove, or fasttext (default: Word2Vec)",
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
        choices=["cxOnePoint", "cx_uniform", "cx_fair", "cx_one", "cx_random"],
        default="cxOnePoint",
        help="The crossover method to use. cxOnePoint, cx_uniform, cx_fair, cx_one, or cx_random (default: cxOnePoint)",
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
        "-d",
        "--debug",
        action="store_true",
        help="Turn on debug prints (default: false)",
    )

    args = parser.parse_args()

    return Config(
        embeddings=args.embeddings,
        dimension=args.dimension,
        population_size=args.population_size,
        crossover_method=args.crossover_method,
        cross_prob=args.prob_crossover,
        mut_prob=args.prob_mutation,
        num_generations=args.generations,
        debug=args.debug,
    )
