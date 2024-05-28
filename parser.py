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