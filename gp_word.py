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
from deap.gp import PrimitiveSet, genGrow
import math


#input(Config.pop_size, Config.dim, Config.cx_method, Config.mut_pb, Config.n_gen)

def protected_div(x, y):
    return x / y if y != 0 else 1

class GP:
    def __init__(self, pop_size, dim, cx_method, mut_pb, n_gen, data, df):
        self.pop_size = pop_size
        self.dim = dim
        self.cx_method = cx_method
        self.mut_pb = mut_pb
        self.n_gen = n_gen
        self.pop = None
        self.data = data
        self.df = df
        # self.fitness_val = None

    def register(self):
        #定義算術表達式的原語集（Primitive Set）
        self.pset = gp.PrimitiveSet("MAIN", 5)
        self.pset.addPrimitive(np.add, 2)
        self.pset.addPrimitive(np.subtract, 2)
        self.pset.addPrimitive(np.multiply, 2)
        self.pset.addPrimitive(protected_div, 2) ##確認一次ok
        self.pset.addPrimitive(np.sqrt, 1)
        self.pset.addPrimitive(np.square, 1)
        #self.pset.renameArguments(ARG0='a', ARG1='b', ARG2='c', ARG3='d', ARG4='e')
        #print("Attributes of gp.PrimitiveSet:", dir(self.pset))
        #創建適應度類和個體類
        creator.create("FitnessMax", base.Fitness, weights=(1,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset) #output不算fitness
        #creator.create("Individual", gp.PrimitiveTree) #output不算fitness
        #初始化工具箱
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=5)
        #self.toolbox.register("individual", creator.Individual, fitness=creator.FitnessMax, expr=self.toolbox.expr) #gene_gen=toolbox.gene_gen, n_genes=n_genes
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr) #gene_gen=toolbox.gene_gen, n_genes=n_genes
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual,n = self.pop_size) #population數ok
        #註冊operators
        self.toolbox.register("select", tools.selTournament, k=2, tournsize=3)
        self.toolbox.register("cx_simple", tools.cxOnePoint)#simple crossover
        #toolbox.register("cx_uniform")  !!!
        #toolbox.register("cx_fair")   !!!
        #toolbox.register("cx_one")   !!!
        #toolbox.register("cx_contxt")   !!!
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr, pset=self.pset)
        self.toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter('height'), max_value=5)) 
        #toolbox.pbs['mutate'] =   !!! ## assign the probability along with registration pb 且取決於內部突變操作的概率控制。
        self.toolbox.register("evaluate", self.evaluate)#
        self.toolbox.register("compile", gep.compile_, pset=self.pset)
        #註冊record工具
        stats = tools.Statistics(key=lambda ind: ind.fitness.values) #!!!ind: ind.fitness.values[0] fitness???
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        #print("reg done!")

    def initialize_pop(self):
        self.register()
        #print(self.pop_size)
        self.pop = self.toolbox.population(n=self.pop_size)
        #print(self.pop)
        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, self.pop)
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit

    def get_data():
        pass

    def evaluate(self, individual):
        """Evalute the fitness of an individual"""
        func = gp.compile(individual, self.pset)
        for (a, b, c, d, e), y in zip(X, Y):
            predict = func(a, b, c, d, e) ###!!!!
            similarity = cosine_similarity(predict, y) ###!!!
        return similarity

    def cx_uniform():
       pass
    def cx_fair():
       pass
    def cx_one():
       pass   
    def cx_contxt():
        pass

    def select_p(self):
        parents = self.toolbox.select(self.pop)
        #parents = map(toolbox.clone, parents)
        childs = list(map(toolbox.clone, parents))
        return parents, childs
    
    def crossover(self, parents):
        for a,b in parents:
            if self.cx_method == 1:
                toolbox.cx_simple(a, b)
                fit_a = toolbox.evaluate(a)
                fit_b = toolbox.evaluate(b)
                if fitness_a <= fitness_b:
                    parents.remove(a)
                else:
                    parents.remove(b)
            elif self.cx_method == 2:
                pass
                #toolbox.cx_uniform
            elif self.cx_method == 3:
                pass
                #toolbox.cx_fair(a, b)
            elif self.cx_method == 4:
                pass
                #toolbox.cx_one(a, b)
            elif self.cx_method == 5:
                pass
                #toolbox.cx_contxt(a, b)
            elif self.cx_method == 6:
                pass
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


         
         
# def GP(Config):
#     gpp = GP(Config.pop_size, Config.dim, Config.cx_method Config.mut_pb, Config.n_gen)
#     gpp.initialize_pop()
#     gpp.evolve()
#     return

def run_GP(pop_size, dim, cx_method, mut_pb, n_gen, data, df):
    #print(df)
    print(data)
    print(len(data))
    x = data[0].str.split(' ').apply(lambda x: x[:5])
    y = data[0].str.split(' ').str.get(5)
    print(x)
    print(y)
    test = y.iloc[0]
    print(test)
    if test in df.index:
        y_embedding = df.loc[test]
        print(y_embedding)
    else:
        print(f"Embedding for '{test}' not found in the dataset.")

    #gpp = GP(pop_size, dim, cx_method, mut_pb, n_gen, dataset, df)
    #gpp.initialize_pop()
    #gpp.evolve()
    return