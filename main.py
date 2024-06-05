from parser import parse_arguments, Config

from gp import run_GP

if __name__ == "__main__":
    config = parse_arguments()

    # using '-d' to turn on debug flag
    if config.debug:
        config.print_configuration()
    run_GP(config)
