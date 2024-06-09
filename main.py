import sys
from parser import parse_arguments, Config

from simple_gp import simple_gp
from rf import random_forest
from gpab import gpab

if __name__ == "__main__":
    config = parse_arguments()
    # print(f"config:{config.dimension}")

    # using '-d' to turn on debug flag
    if config.debug:
        config.print_configuration()

    if config.algorithm == "simple_gp":
        simple_gp(config)
    elif config.algorithm == "rf":
        random_forest(config)
    else:  # gp_type == "gpab":
        gpab(config)
    # elif gp_type == "rf":
    #     rf(config)
    # else:  # gp_type == "gpab":
    #     gpab(config)
