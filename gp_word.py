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

#input(Config.pop_size, Config.dim, Config.cx_method, Config.mut_pb, Config.n_gen)

class GP:
    def __init__(self, pop_size, dim, cx_method, mut_pb, n_gen):
        self.pop_size = pop_size
        self.dim = dim
        self.cx_method = cx_method
        self.mut_pb = mut_pb
        self.n_gen = n_gen
        self.pop = None
        # self.fitness_val = None
    
    def register(individual):
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

    def initialize_pop(self):
        register()
        self.pop = toolbox.population(self.pop_size)

         # Evaluate the entire population
        fitnesses = map(toolbox.evaluate, self.pop)
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit

    def get_data():
        pass

    def evaluate(individual):
        """Evalute the fitness of an individual"""
        func = toolbox.compile(individual)
        for (a, b, c, d, e), y in zip(X, Y):
            predict = func(a, b, c, d, e) ###!!!!
            similarity = cosine_similarity(predict, y) ###!!!
        return similarity
    
    def cx_uniform():

    def cx_fair():

    def cx_one():
        
    def cx_contxt():


    
    def select_p(self):
        parents = toolbox.select(self.pop)
        #parents = map(toolbox.clone, parents)
        childs = list(map(toolbox.clone, parents))
        return parents, childs
    
    def crossover(self, parents):
        for a,b in parents:
            if self.cx_method == 1:
                toolbox.cx_simple(a, b)
                fit_a = toolbox.evaluate(a)
                fit_b = toolbox.evaluate(b)
                if fitness_a < fitness_b:
                    self.pop.remove(a)
                else:
                    self.pop.remove(b)
            if self.cx_method == 2:
                toolbox.cx_uniform
            if self.cx_method == 3:
                toolbox.cx_fair(a, b)
            if self.cx_method == 4:
                toolbox.cx_one(a, b)
            if self.cx_method == 5:
                toolbox.cx_contxt(a, b)
            if self.cx_method == 6:
                #未完成!!!!!
        
        return child
    
    def mutate(self, child):
        if random.random() < self.mut_pb:
            toolbox.mutate(child)
            del child.fitness.values
        return child
    
    def select_s(self, parents, child):
        all_individuals = parents + [child]
        all_individuals.sort(key=lambda ind: toolbox.evaluate(ind))
        min_individual = all_individuals[0]
        if min_individual == child:
            return parents
        else:
            for i in range(len(parents)):
                if parents[i] == min_individual:
                    parents[i] = child
                    break
        return parents

    
    def evolve():
        for g in range(self.n_gen):
            parents, childs = self.select_p()
            child = self.crossover(childs)
            child = self.mutate(child)
            self.select_s(parents, child)


         
         
def GP(Config):
    gpp = GP(Config.pop_size, Config.dim, Config.cx_method Config.mut_pb, Config.n_gen)
    gpp.initialize_pop()
    gpp.evolve()
    return