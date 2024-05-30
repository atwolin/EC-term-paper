import argparse

class Config:
    def __init__(self, dimension, representation, population_size, uniform_crossover, cross_prob, mut_prob, num_generations, debug):
        self.dimension = dimension
        self.representation = representation
        self.population_size = population_size
        self.uniform_crossover = uniform_crossover
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob
        self.num_generations = num_generations
        self.crossover_method = self.determine_crossover_method()
        self.debug = debug

    def determine_crossover_method(self):
        if not self.uniform_crossover:
            if self.representation == 'binary':
                return '2-point'
            elif self.representation == 'real':
                return 'whole arithmetic'
        return 'uniform'

    def print_configuration(self):
        parameters = {
            'dimension': self.dimension,
            'representation': self.representation,
            'population_size': self.population_size,
            'uniform_crossover': self.uniform_crossover,
            'crossover_method': self.crossover_method,
            'cross_prob': self.cross_prob,
            'mut_prob': self.mut_prob,
            'num_generations': self.num_generations,
        }
        
        print('-' * 44)
        print('|{:<20}|{:<20}|'.format('Parameter', 'Value'))
        print('-' * 44)
        for key, value in parameters.items():
            print('|{:<20}|{:<20}|'.format(key, str(value)))
        print('-' * 44)

def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments():
    parser = argparse.ArgumentParser(description="GA Configuration Parser")
    parser.add_argument('-n', '--dimension', type=int, default=10, help='The dimension of Schwefel function (default: 10)')
    parser.add_argument('-r', '--representation', choices=['binary', 'real'], default='binary', help='The representation to use. Binary or real-valued (default: binary)')
    parser.add_argument('-p', '--population_size', type=int, default=100, help='Number of the population (default: 100)')
    parser.add_argument('-u', '--uniform_crossover', type=str_to_bool, default=False, help='The crossover method using uniform crossover (1) or not (0). If not, then for binary GA, it will use 2-point crossover and for real-valued GA will use whole arithmetic crossover (default: 0)')
    parser.add_argument('-c', '--pc', type=float, default=0.9, help='Probability for the crossover (default: 0.9)')
    parser.add_argument('-m', '--pm', type=float, default=0.1, help='Probability for the mutation (default: 0.1)')
    parser.add_argument('-g', '--generations', type=int, default=500, help='Max number of generations to terminate (default: 500)')
    parser.add_argument('-d', '--debug', action='store_true', help='Turn on debug prints (default: false)')

    args = parser.parse_args()
    
    return Config(dimension=args.dimension, representation=args.representation, population_size=args.population_size, uniform_crossover=args.uniform_crossover, cross_prob=args.pc, mut_prob=args.pm, num_generations=args.generations, debug=args.debug)