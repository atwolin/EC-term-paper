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
import random
import copy



#input(Config.pop_size, Config.dim, Config.cx_method, Config.mut_pb, Config.n_gen)

# def protected_div(x, y):
#     return x / y if y != 0 else 1
def protected_div(x, y):
    mask = (y == 0)
    safe_y = np.where(mask, 1, y)
    return np.where(mask, 1, x / safe_y)

def protected_sqrt(x):
    x = np.abs(x)
    return np.sqrt(x)

class GP:
    def __init__(self, pop_size, dim, cx_method, mut_pb, n_gen, data, embeddings, x, y):
        self.pop_size = pop_size
        self.dim = dim
        self.cx_method = cx_method
        self.mut_pb = mut_pb
        self.n_gen = n_gen
        self.pop = None
        self.data = data
        self.embeddings = embeddings
        self.inputword = x
        self.realword = y
        # self.fitness_val = None

    def register(self):
        #定義算術表達式的原語集（Primitive Set）
        self.pset = gp.PrimitiveSet("MAIN", 5)
        self.pset.addPrimitive(np.add, 2)
        self.pset.addPrimitive(np.subtract, 2)
        self.pset.addPrimitive(np.multiply, 2)
        self.pset.addPrimitive(protected_div, 2) ##確認一次ok
        #self.pset.addPrimitive(np.sqrt, 1)
        self.pset.addPrimitive(protected_sqrt, 1)
        self.pset.addPrimitive(np.square, 1)
        self.pset.renameArguments(ARG0='a', ARG1='b', ARG2='c', ARG3='d', ARG4='e')
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
        self.toolbox.register("cx_simple", gp.cxOnePoint)#simple crossover
        #self.toolbox.register("cx_simple", gp.cxOnePointLeafBiased)#simple crossover
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
        #for ind in self.pop:
        #    print(str(ind))
        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, self.pop)
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit

    def get_data():
        pass

    # def get_embedding(self, word):
    #     print(word)
    #     if word in self.embeddings.index:
    #         embedding = self.embeddings.loc[word]
    #         print(embedding)
    #     else:
    #         print(f"Embedding for '{word}' not found in the dataset.")
    #         print(self.embeddings)

    # def get_embedding(self, word):
    #     #print(word)
    #     if word in self.embeddings.index:
    #         embedding = self.embeddings.loc[word]
    #         #print(embedding)
    #     else:
    #         #print(f"Embedding for '{word}' not found in the dataset.")
    #         #print(self.embeddings)
    #         embedding = np.full(10, 0.5)
    #     return embedding

    def clean_data(self, data):
        data = np.where(np.isinf(data), np.finfo(np.float32).max, data)
        data = np.nan_to_num(data, nan=0.0)
        return data

    def evaluate(self, individual):
        """Evalute the fitness of an individual"""
        func = gp.compile(individual, self.pset)
        total_similarity = 0.0
        for data_index in range(len(self.inputword)):
            words = self.inputword.iloc[data_index]
            in_vectors = [self.embeddings[word] for word in words]
            a, b, c, d, e = in_vectors[:5]
            # has_nan_in = np.isnan(in_vectors).any()
            # if has_nan_in:
            #      print("in_vector 中有元素为 nan")
            # if any(vector is None for vector in [a, b, c, d, e]):
            #     print(f"Skipping index {data_index} due to None values in vectors: {vector}")
            # else:
            #     print("in沒有0")
            # print("檢查點")
            y = self.realword.iloc[data_index]
            out_vector = self.embeddings[y]
            # has_nan_out_vector = np.isnan(out_vector).any()
            # if has_nan_out_vector:
            #      print("out_vector 中有元素为 nan")
            # if has_zero:
            #     print("out_vector 中有元素为 0")
            # else:
            #     print("out_vector 中没有元素为 0")
            # #print(f"y的embedding:{out_vector}")
            predict = self.clean_data(func(a, b, c, d, e))
            # has_nan_predict = np.isnan(predict).any()
            # if has_nan_predict:
            #      print("predict 中有元素为 nan")
            # else:
            #     print("predict 中没有元素为 0")
            # #similarity = cosine_similarity(predict, y) ###!!!
            
            similarity = cosine_similarity([predict], [out_vector])[0][0]
            total_similarity += similarity
        fitness = total_similarity / len(self.inputword)
        ftiness = self.clean_data(fitness)
        return (fitness, )
        
    # def evaluate(self, individual):
    # #"""Evaluate the fitness of an individual"""
    # # 编译个体以获取其表示的函数
    #     func = gp.compile(individual, pset=self.pset)
    #     total_similarity = 0.0
    #     #print(f"self.inputword:{self.inputword}")
    #     #print(f"len(self.inputword)::{len(self.inputword)}")
    #     for data_index in range(len(self.inputword)):
    #         #print(f"data_index:{data_index}")
    #         words = self.inputword.iloc[data_index]
    #         #print(f"words: {words}")
    #         vectors = [self.get_embedding(word) for word in words]
    #         #print(f"vectors: {vectors}")
    #         # 获取输入X的5个10维向量
    #         a, b, c, d, e = vectors[:5]
    #         if any(vector is None for vector in [a, b, c, d, e]):
    #             print(f"Skipping index {data_index} due to None values in vectors: {vector}")
    #             continue
    #         # 获取对应的Y值
    #         #print(f"data_index:{data_index}")
    #         y = self.realword.iloc[data_index]
    #         vector = self.get_embedding(y)
    #         #print(f"realword & vector : {y} , {vector}")
    #         # 计算预测值
    #         predict = func(a, b, c, d, e)
    #         # 计算预测值和真实值的余弦相似度
    #         #similarity = cosine_similarity([predict], [vector])[0][0]
    #         #total_similarity += similarity

    #         # 返回平均相似度作为适应度
    #     #zzz = total_similarity / len(self.inputword)
    #     #print(f"平均相似度 & type:{zzz} {type(zzz)}")
    #     #print
    #     zzz = 0.19
    #     return (zzz,)


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
        childs = copy.deepcopy(parents)
        return parents, childs
    
    def crossover(self, parents):
        #print(f"父母為：{parents}")
        parent1, parent2 = parents
        #print(f"A是：{parent1}")
        #print(f"B是：{parent2}")
        
        if self.cx_method == 1:
            a,b = self.toolbox.cx_simple(parent1, parent2)
            #print(f("parents, childs是＿和＿tpye： {parents} ,{childs} ; {type(parents)},{type(childs)}"))
            fit_a = self.toolbox.evaluate(a)
            fit_b = self.toolbox.evaluate(b)
            if fit_a <= fit_b:
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

        return parents
    
    def mutate(self, child):
        if random.random() < self.mut_pb:
            self.toolbox.mutate(child)
            del child.fitness.values
        return child
    
    def select_s(self, parents, child):
        #print(f"父母：{parents}")
        #print(f"子代：{child}")
        all_individuals = parents + child
        #print(f"所有：{all_individuals}")
        all_individuals.sort(key=lambda ind: self.toolbox.evaluate(ind))
        min_individual = all_individuals[0]
        if min_individual == child:
            return parents
        else:
            for i in range(len(parents)):
                if parents[i] == min_individual:
                    parents[i] = child
                    break
        return parents

    
    def evolving(self):
        for g in range(self.n_gen):
            parents, childs = self.select_p()
            #print(f"父母類型： {type(parents)} 小孩類型：{type(childs)}：parents是： {parents}  childs是：{childs}")
            child = self.crossover(childs)
            child = self.mutate(child)
            self.select_s(parents, child)


         
         
# def GP(Config):
#     gpp = GP(Config.pop_size, Config.dim, Config.cx_method Config.mut_pb, Config.n_gen)
#     gpp.initialize_pop()
#     gpp.evolve()
#     return

def run_GP(pop_size, dim, cx_method, mut_pb, n_gen, data, embeddings):
    #print(embeddings)
    #print(data)
    #print(len(data))

    x = data[0].str.split(' ').apply(lambda x: x[:5])
    y = data[0].str.split(' ').str.get(5)

    # missing_words = []
    # for sentence in x:
    #     #print(sentence)
    #     for word in sentence:
    #         if word not in embeddings.index:
    #             missing_words.append(word)
    #             #print(f"Word '{word}' not found in embeddings")
    # print(f"Total missing words: {len(missing_words)}")
    # print(f"Missing words: {missing_words}")

    # print(x)
    # print(y)
    # # test = y.iloc[0]
    # print(test)
    # if test in embeddings.index:
    #     y_embedding = embeddings.loc[test]
    #     print(y_embedding)
    # else:
    #     print(f"Embedding for '{test}' not found in the dataset.")

    gpp = GP(pop_size, dim, cx_method, mut_pb, n_gen, data, embeddings, x, y)
    gpp.initialize_pop()
    gpp.evolving()
    return