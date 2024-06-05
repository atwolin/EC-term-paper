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
#from collections import defaultdict



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
        self.toolbox.register("cx_uniform", self.cx_uniform) 
        self.toolbox.register("cx_fair", self.cx_fair)   
        self.toolbox.register("cx_one", self.cxOnePoint)   
        
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
        print(f"selfpop種類:{type(self.pop)}")

    def subtree_height(self, tree, index):
    #"""Calculate the height of the subtree starting at the given index."""
        def _height(node_index):
            node = tree[node_index]
            if node.arity == 0:  # Leaf node
                return 1
            else:
                return 1 + max(_height(child_index) for child_index in range(node_index + 1, node_index + 1 + node.arity))    
        return _height(index)

    def searchSubtree_idx(self, tree, begin):
        end = begin + 1
        total = tree[begin].arity
        while total > 0:
            total += tree[end].arity - 1
            end += 1
        return begin, end
    
    def searchSubtree(self, begin):
        end = begin + 1
        total = self[begin].arity
        while total > 0:
            total += self[end].arity - 1
            end += 1
        return slice(begin, end)

    def clean_data(self, data):
        data = np.where(np.isinf(data), np.finfo(np.float32).max, data)
        data = np.nan_to_num(data, nan=0.0)
        return data

    def evaluate(self, individual):
        """Evalute the fitness of an individual"""
        #print(f"individual種類:{type(individual)}")
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
        


    def cx_uniform(self, ind1, ind2):

        if len(ind1) < 2 or len(ind2) < 2:
            # No crossover on single node tree
            return ind1, ind2
        # if len(ind1) >= len(ind2):
        #     parent = ind1
        # else:
        #     parent = ind2
        #child1 = creator.Individual(ind1)
        #child2 = creator.Individual(ind2)
        child = type(ind1)([])
        parents = [ind1, ind2]
        flag0, flag1 = 0, 0
        # p0 = parents[0].searchSubtree(0)
        # p1 = parents[1].searchSubtree(0)
        left_0 = parents[0].searchSubtree(1)            
        left_1 = parents[1].searchSubtree(1)
        b0, e0 = self.searchSubtree_idx(parents[0],1)
        #print(f"b0={b0}, e0={e0}")
        #print(f"parents[0]:{len(parents[0])}")
        b1, e1 = self.searchSubtree_idx(parents[1],1)
        #print(f"b1={b1}, e1={e1}")
        #print(f"parents[1]:{len(parents[1])}")
        if e0+1 < len(parents[0]):
            right_0 = parents[0].searchSubtree(e0+1)
            flag0=1
        if e1+1 < len(parents[1]):
            right_1 = parents[1].searchSubtree(e1+1)
            flag1=1
        left = [left_0, left_1]
        if flag0 == 1 and flag1 == 1:
            right = [right_0, right_1]
        #print(f"left: {left}")
        #print(f"right: {right}")
        r = random.randint(0, 1) #r是root
        if flag0 == 1 and flag1 == 1:
            r1 = random.randint(0, 1) #r1是左邊
        #print(f"r={r}, r1={r1}")
        #print(f"第一個：{parents[r][left[r]]}/{parents[r1][left[r1]]}")
            parents[r][left[r]] = parents[r1][left[r1]]
            r2 = random.randint(0, 1)
            parents[r][right[r]] = parents[r2][right[r2]]
        else:
            #print("只有一個子點")
            r1 = random.randint(0, 1)
            parents[r][left[r1]] = parents[r1][left[r1]]
        # print("告一段落")
        #print(parents[0])
        return parents[0], parents[0]
       
    def cx_fair(self, ind1, ind2):
    # """size fair crossover for two trees.
    # :param ind1: First tree participating in the crossover.
    # :param ind2: Second tree participating in the crossover.
    # :returns: A tuple of two trees.
    # """
        if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
            return ind1, ind2

        # List all available primitive types in each individual
        types1 = gp.defaultdict(list)
        types2 = gp.defaultdict(list)
        if ind1.root.ret == gp.__type__:
            # Not STGP optimization
            types1[gp.__type__] = list(range(1, len(ind1)))
            types2[gp.__type__] = list(range(1, len(ind2)))
            common_types = [gp.__type__]
        else:
            for idx, node in enumerate(ind1[1:], 1):
                types1[node.ret].append(idx)
            for idx, node in enumerate(ind2[1:], 1):
                types2[node.ret].append(idx)
            common_types = set(types1.keys()).intersection(set(types2.keys()))

        if len(common_types) > 0:
            type_ = random.choice(list(common_types))

        index1 = random.choice(types1[type_])
        height1 = self.subtree_height(ind1, index1)
        #height = ind1.height
        
        while(1):
            index2 = random.choice(types2[type_])
            height2 = self.subtree_height(ind2, index2)
            if height2 <= height1:
                #print(f"height1: {height1}, height2: {height2}")
                break 
        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

        #print(f"Parent 1:{types1}")
        #print(f"Parent 1:{types1[3]}")
        return ind1, ind2
    
    def combine_child(self, stack):
        while (len(stack[-1][1]) == stack[-1][0].arity and len(stack[-1][1]) < 2):
        # print(f"len(stack[-1][1]={len(stack[-1][1])}, stack[-1][0].arity={stack[-1][0].arity}")
        # Extract child
            prim, args, i2 = stack.pop()
            string = prim.format(*args)
            if len(stack) == 0:
                break
        # Add to its parent
            stack[-1][1].append(string)
        # print(f"[UPDATE] append stack: {stack[-1][1]}")
        # print(f"stack: {stack}")


    def traverse_tree(self, stack, res, parent, idx):
        while (res != 0):
        # arity1 += 1
        # print(f"arity1: {arity1}")

            res -= 1
        # print(f"[WHILE -1] res: {res}")

            idx += 1
            stack.append((parent[idx], [], idx))
        # print(f"[WHILE] append stack: {parent[idx1].name}")
            res += parent[idx].arity
        # print(f"[WHILE +arity] res: {res}")

        # print("combine")
            self.combine_child(stack)

        # print(f"stack: {stack}")
        return stack, res, idx

    def cxOnePoint(self, ind1, ind2):
   
        idx1 = 0
        idx2 = 0
    # To track the trees
        stack1 = []
        stack2 = []
    # Store the common region
        region1 = []
        region2 = []

    # Start traversing the trees
        while (idx1 < len(ind1) and idx2 < len(ind2)):
        # Push the nodes to the stack
        # print("================================NEW================================")
            stack1.append((ind1[idx1], [], idx1))
            stack2.append((ind2[idx2], [], idx2))
        # print(f"append stack1: {ind1[idx1].name}")
        # print(f"append stack2: {ind2[idx2].name}")
        # print(f"stack1: {stack1}")
        # print(f"stack2: {stack2}")


        # Not the same region
            if (stack1[-1][0].arity != stack2[-1][0].arity):
                res1 = stack1[-1][0].arity
                res2 = stack2[-1][0].arity
            # print(f"res1: {res1}, res2: {res2}")
            # arity1 = 0  # number of child nodes of the current node
            # arity2 = 0
                stack1, res1, idx1 = self.traverse_tree(stack1, res1, ind1, idx1)
            # print("----------------------------------------STACK 2----------------------------------------")
                stack2, res2, idx2 = self.traverse_tree(stack2, res2, ind2, idx2)
            else:
                region1.append([ind1[idx1], idx1])
                region2.append([ind2[idx2], idx2])

        # print("-------------1 loop for combine------------")
            self.combine_child(stack1)
            self.combine_child(stack2)
            idx1 += 1
            idx2 += 1

        #for pri, idx in region1:
            #print(f"{idx}: {pri.name}")

    # Select crossover point
        point = random.randint(0, len(region1) - 1)
        #print(f"crossover point: {point}")
        #print(f"crossover point for trees: {region1[point]}, {region2[point]}")

    # Swap subtrees
        if (len(region1) > 0):
            slice1 = ind1.searchSubtree(region1[point][1])
            slice2 = ind2.searchSubtree(region2[point][1])
            ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    # Select the one has higher fitness value
    ### TODO ###

        return ind1, ind2

    def select_p(self):

        parents = self.toolbox.select(self.pop)
        #print(f"parents類型：{type(parents)}")
        #parents = map(toolbox.clone, parents)
        childs = copy.deepcopy(parents)
        return parents, childs
    
    def crossover(self, parents):
        #print(f"父母為：{parents}")
        parent1, parent2 = parents
        #print(f"A是：{parent1}")
        #print(f"B是：{parent2}")
        
        if self.cx_method == 5:
            choice = random.choice([self.toolbox.cx_simple, self.toolbox.cx_uniform, self.toolbox.cx_fair, self.toolbox.cx_one])
            a,b = choice(parent1, parent2)
            print(f"choice:{choice}")
        if self.cx_method == 1:
            a,b = self.toolbox.cx_simple(parent1, parent2)
            #print(f("parents, childs是＿和＿tpye： {parents} ,{childs} ; {type(parents)},{type(childs)}"))
            # fit_a = self.toolbox.evaluate(a)
            # fit_b = self.toolbox.evaluate(b)
            # if fit_a <= fit_b:
            #     parents.remove(a)
            # else:
            #     parents.remove(b)
        if self.cx_method == 2:
            a = self.toolbox.cx_uniform(parent1, parent2)
            #toolbox.cx_uniform
        if self.cx_method == 3:
            a,b = self.toolbox.cx_fair(parent1, parent2)
            #toolbox.cx_fair(a, b)
        if self.cx_method == 4:
            a,b = self.toolbox.cx_one(parent1, parent2)
                #toolbox.cx_one(a, b)
        # elif self.cx_method == 5: #random
        #     pass
                #未完成!!!!!
        fit_a = self.toolbox.evaluate(a)
        if self.cx_method == 2:
            return a
        fit_b = self.toolbox.evaluate(b)
        if fit_a <= fit_b:
            parents.remove(a)
        else:
            parents.remove(b)
            

        return parents
    
    def mutate(self, child):
        #print(f"mutate類別：{type(child)}")
        #print(child)
        if random.random() < self.mut_pb:
            print("mutate開始！")
            self.toolbox.mutate(child)
            del child.fitness.values
        #evaluate = self.toolbox.evaluate(child[0])
        #print("mutate完成！")
        return child
    
    def select_s(self, parents, child):
        #print(f"父母：{parents}")
        #print(f"子代：{child}")
        c_f = self.toolbox.evaluate(child[0])
        #c_f = child.fitness.valuesx
        p0_f = parents[0].fitness.values
        p1_f = parents[1].fitness.values    
        if c_f <= p0_f and c_f <= p1_f:
            #print(f"最小的适应度值: {min_f} 对应的个体: {child}")
            return
        else:
            if min(p0_f, p1_f) == p0_f:
                parents.remove(parents[0])
                parents.append(child)
            else:
                parents.remove(parents[1])
                parents.append(child)
        return
    
    def evolving(self):
        for g in range(self.n_gen):
            parents, childs = self.select_p()
            #print(f"父母類型： {type(parents)} 小孩類型：{type(childs)}")
            child = self.crossover(childs)
            #print(f"交叉完成type！: {type(child)}")
            child = self.mutate(child)
            #print(f"突變完成type！: {type(child)}")
            self.select_s(parents, child)
            #self.pop.append()
            print(f"第{g}代完成！")


         
         
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