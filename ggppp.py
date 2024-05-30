import fasttext
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import geppy as gep
from deap import creator, base, tools, algorithms
import operator
import deap.gp as gp

#input(Config.pop_size, Config.dim, Config.cx_method Config.mut_pb, Config.n_gen)

##initial/parentsel/cross/mut/survive/evaluation
#/breeding/survivor selection/print best/run trail

def ga(Config):
    get_data = 
    #input(Config.pop_size, Config.dim, Config.cx_method Config.mut_pb, Config.n_gen)

    # if Config.representation == 'binary':

    #     binary_ga = BinaryGA(Config.population_size, Config.dimension, Config.cross_prob, Config.mut_prob, Config.uniform_crossover)
    #     binary_ga.initialize_population()
    #     binary_ga.run(generations=Config.num_generations)

    # else:
    #     # 初始化實數遺傳算法
    #     real_valued_ga = RealValuedGA(Config.population_size, Config.dimension, Config.cross_prob, Config.mut_prob, Config.uniform_crossover)
    #     real_valued_ga.initialize_population()
    #     real_valued_ga.run(generations=Config.num_generations)
    # return
    # #....


def regester(individual):
    #定義算術表達式的原語集（Primitive Set）
    pset = gep.PrimitiveSet('main', input_names=['word1', 'word2', 'word3', 'word4', 'word5'])
    pset.add_function(np.add, 2)
    pset.add_function(np.subtract, 2)
    pset.add_function(np.multiply, 2)
    pset.add_function(np.divide, 2) ##確認一次
    pset.add_function(np.sqrt, 1)
    pset.add_function(np.square, 1)
    #創建適應度類和個體類
    creator.create("FitnessMax", base.Fitness, weights=(1,))
    creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMax) #output不算fitness
    #初始化工具箱
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf(), pset=pset, min_=1, max_=5)
    toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes, linker=operator.add) #gene_gen=toolbox.gene_gen, n_genes=n_genes
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) #population數
    #註冊operators
    #toolbox.register("steady_select", tools.selBest, k=2, ???) #!!!deap.tools.selBest(individuals, k, fit_attr='fitness')[source]
    toolbox.register("select", tools.selTournament, k=2, tournsize=3)
    toolbox.register("cx_simple", tools.cxOnePoint)#simple crossover
    #toolbox.register("cx_uniform")  !!!
    #toolbox.register("cx_fair")   !!!
    #toolbox.register("cx_one")   !!!
    #toolbox.register("cx_contxt")   !!!
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter('height'), max_value=5)) 
    #toolbox.pbs['mutate'] =   !!! ## assign the probability along with registration pb 且取決於內部突變操作的概率控制。
    toolbox.register("evaluate", evaluate)#
    toolbox.register("compile", gep.compile_, pset=pset)
    #註冊record工具
    stats = tools.Statistics(key=lambda ind: ind.fitness.values) #!!!ind: ind.fitness.values[0] fitness???
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

def initialize(Config.pop_size, Config.mut_pb, Config.n_gen):
    pop = toolbox.population(n=50)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40 ##!!!!


#Operators
def evaluate(individual):
    """Evalute the fitness of an individual"""
    # inserting x and y into func and
    # compute the fitness of this individual
    # ....
    func = toolbox.compile(individual)
    for (a, b, c, d, e), y in zip(X, Y):
        predict = func(a, b, c, d, e) ###!!!!
        similarity = cosine_similarity(predict, y) ###!!!
    return similarity

def get_data():
    # generate the training set
    data = #。。。！！！
    X = data[:, :5]
    Y = data[:, -1]
    data_pairs = zip(X, Y) #tensor numpy

def sel_parent():
    

    

##_________________________舊版以複製##

# #定義算術表達式的原語集（Primitive Set）
# pset = gep.PrimitiveSet('main', input_names=['word1', 'word2', 'word3', 'word4', 'word5'])
# pset.add_function(np.add, 2)
# pset.add_function(np.subtract, 2)
# pset.add_function(np.multiply, 2)
# pset.add_function(np.divide, 2) ##確認一次
# pset.add_function(np.sqrt, 1)
# pset.add_function(np.square, 1)

# #創建適應度類和個體類
# creator.create("FitnessMax", base.Fitness, weights=(1,))
# creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMax) #output不算fitness

# #初始化工具箱
# toolbox = base.Toolbox()
# toolbox.register("expr", gp.genHalfAndHalf(), pset=pset, min_=1, max_=5)
# toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes, linker=operator.add) #gene_gen=toolbox.gene_gen, n_genes=n_genes
# toolbox.register("population", tools.initRepeat, list, toolbox.individual) #population數

# #Operators

# def evaluate(individual):
#     """Evalute the fitness of an individual"""
#     # inserting x and y into func and
#     # compute the fitness of this individual
#     # ....
#     func = toolbox.compile(individual)
#     for (a, b, c, d, e), y in zip(X, Y):
#         predict = func(a, b, c, d, e) ###!!!!
#         similarity = cosine_similarity(predict, y) ###!!!
#     return similarity



# # Register operators
# #toolbox.register("steady_select", tools.selBest, k=2, ???) #!!!deap.tools.selBest(individuals, k, fit_attr='fitness')[source]
# toolbox.register("select", tools.selTournament, k=2, tournsize=3)
# toolbox.register("cx_simple", tools.cxOnePoint)#simple crossover
# #toolbox.register("cx_uniform")  !!!
# #toolbox.register("cx_fair")   !!!
# #toolbox.register("cx_one")   !!!
# #toolbox.register("cx_contxt")   !!!
# toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
# toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter('height'), max_value=5)) 
# #toolbox.pbs['mutate'] =   !!! ## assign the probability along with registration pb 且取決於內部突變操作的概率控制。
# toolbox.register("evaluate", evaluate)#
# toolbox.register("compile", gep.compile_, pset=pset)



# # generate the training set
# data = #。。。！！！
# X = data[:, :5]
# Y = data[:, -1]
# data_pairs = zip(X, Y) #tensor numpy

# stats = tools.Statistics(key=lambda ind: ind.fitness.values) #!!!ind: ind.fitness.values[0] fitness???

# stats.register("avg", numpy.mean)
# stats.register("std", numpy.std)
# stats.register("min", numpy.min)
# stats.register("max", numpy.max)

# size of population and number of generations
n_pop = 500
n_gen = 30

pop = toolbox.population(n=n_pop)
hof = tools.HallOfFame(1)   # only record the best individual ever found in all generations

# start evolution
#pop, log = gep.gep_simple(pop, toolbox, n_generations=n_gen, n_elites=2, stats=stats, hall_of_fame=hof, verbose=True)

def main():
    pop = toolbox.population(n=50)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40 ##!!!!

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop)
        # Clone the selected individuals
        offspring = map(toolbox.clone, offspring)
        parent = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                # crossover mode selection ！！！！！
                if mode = 1:
                    toolbox.cx_simple(child1, child2)


                del child1.fitness.values
                del child2.fitness.values
                #child 先選一個

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    return pop
