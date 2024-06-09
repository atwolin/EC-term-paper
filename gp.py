import os
import re
import operator
import random
import copy
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from deap import creator, base, tools, algorithms
import deap.gp as gp
from deap.gp import PrimitiveSet, genGrow
from deap.gp import cxOnePoint as cx_simple
from data import get_embeddings

cur_path = os.getcwd()
PATH = re.search(r"(.*EC-term-paper)", cur_path).group(0)
# print(PATH)

CX_RANDOM = 0
CX_SIMPLE = 1
CX_UNIFORM = 2
CX_FAIR = 3
CX_ONE_POINT = 4


def get_cx_num(crossover_method):
    if crossover_method == "cx_random":
        cx_num = CX_RANDOM
    elif crossover_method == "cx_simple":
        cx_num = CX_SIMPLE
    elif crossover_method == "cx_uniform":
        cx_num = CX_UNIFORM
    elif crossover_method == "cx_fair":
        cx_num = CX_FAIR
    else:  # crossover_method == "cx_one_point"
        cx_num = CX_ONE_POINT
    return cx_num


def protected_div(x, y):
    mask = y == 0
    safe_y = np.where(mask, 1, y)
    return np.where(mask, 1, x / safe_y)


def protected_sqrt(x):
    x = np.abs(x)
    return np.sqrt(x)


def get_X(trees):
    X_list = np.array(
        [
            np.array([trees.embeddings[char] for char in words])
            for words in trees.inputword
        ]
    )
    return X_list


def get_predict_vec(trees, individual):
    func = gp.compile(individual, trees.pset)
    y_pred_list = np.array(
        [
            func(*np.array([trees.embeddings[char] for char in words]))
            for words in trees.inputword
        ]
    )
    return y_pred_list


def get_y(trees):
    y_true_list = np.array(
        np.array([trees.embeddings[char] for char in trees.realword])
    )
    return y_true_list


def get_predict_word(y_pred_vec, embedding_type, embedding_model):
    if embedding_type == "word2vec":
        y_pred_word = embedding_model.wv.similar_by_vector(y_pred_vec, topn=1)
        return y_pred_word[0][0]
    elif embedding_type == "glove":
        y_pred_word = embedding_model.similar_by_vector(y_pred_vec, topn=1)
        return y_pred_word[0][0]
    else:  # embedding_type == "fasttext"
        y_pred_word = embedding_model.get_nearest_neighbors(y_pred_vec)
        return y_pred_word[0][1]


class GP:
    def __init__(
        self,
        algorithm,
        embedding_type,
        dimension,
        population_size,
        crossover_method,
        cross_prob,
        mut_prob,
        num_generations,
        num_evaluations,
        data,
        embeddings,
        run,
    ):
        self.algorithm = algorithm
        self.embedding_type = embedding_type
        self.dim = dimension
        self.pop_size = population_size
        self.cx_method = crossover_method
        self.cx_pb = cross_prob
        self.mut_pb = mut_prob
        self.max_gen = num_generations
        self.max_eval = num_evaluations
        self.data = data
        self.inputword = data[0].str.split(" ").apply(lambda x: x[:5])
        self.realword = data[0].str.split(" ").str.get(5)
        self.embeddings = embeddings
        self.run = run

        self.pop = None
        self.n_gen = 0
        self.eval_count = 0

    def register(self):
        # Function set
        self.pset = gp.PrimitiveSet("MAIN", 5)
        self.pset.addPrimitive(np.add, 2)
        self.pset.addPrimitive(np.subtract, 2)
        self.pset.addPrimitive(np.multiply, 2)
        self.pset.addPrimitive(protected_div, 2)
        self.pset.addPrimitive(protected_sqrt, 1)
        self.pset.addPrimitive(np.square, 1)

        # Terminal set
        self.pset.renameArguments(ARG0="a", ARG1="b", ARG2="c", ARG3="d", ARG4="e")

        # Create the fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1,))
        creator.create(
            "Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset
        )

        # Initialize the toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=5)
        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.toolbox.expr
        )
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual,
            n=self.pop_size,
        )
        self.toolbox.register("select", tools.selRandom, k=3)
        self.toolbox.register("cx_simple", cx_simple)
        self.toolbox.register("cx_uniform", self.cx_uniform)
        self.toolbox.register("cx_fair", self.cx_fair)
        self.toolbox.register("cx_one", self.cxOnePoint)

        self.toolbox.register(
            "mutate", gp.mutUniform, expr=self.toolbox.expr, pset=self.pset
        )
        self.toolbox.decorate(
            "mutate", gp.staticLimit(operator.attrgetter("height"), max_value=5)
        )
        self.toolbox.register("evaluate", self.evaluate)  #

        # Record for analysis
        self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        self.hof = tools.HallOfFame(10)  # hall of fame size

    def initialize_pop(self):
        if "FitnessMax" in creator.__dict__:
            del creator.FitnessMax
        if "Individual" in creator.__dict__:
            del creator.Individual
        self.register()
        self.pop = self.toolbox.population(n=self.pop_size)
        fitnesses = map(self.toolbox.evaluate, self.pop)
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit

    def subtree_height(self, tree, index):
        """Calculate the height of the subtree starting at the given index."""

        def _height(node_index):
            node = tree[node_index]
            if node.arity == 0:  # Leaf node
                return 1
            else:
                return 1 + max(
                    _height(child_index)
                    for child_index in range(
                        node_index + 1, node_index + 1 + node.arity
                    )
                )

        return _height(index)

    def searchSubtree_idx(self, tree, begin):
        end = begin + 1
        total = tree[begin].arity
        while total > 0:
            total += tree[end].arity - 1
            end += 1
        return begin, end

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
            # print(f"in_vectors: {in_vectors}")
            y = self.realword.iloc[data_index]
            # print(f"y: {y}")
            out_vector = self.embeddings[y]
            # print(f"out_vector: {out_vector}")
            # if out_vector.ndim == 3:
            # out_vector = out_vector.reshape(1, -1)

            predict = self.clean_data(func(a, b, c, d, e))

            similarity = cosine_similarity([predict], [out_vector])[0][0]
            total_similarity += similarity
        fitness = total_similarity / len(self.inputword)
        ftiness = self.clean_data(fitness)
        self.eval_count += 1
        return (fitness,)

    def cx_uniform(self, ind1, ind2):
        child = type(ind1)([])
        parents = [ind1, ind2]
        flag0, flag1 = 0, 0
        # p0 = parents[0].searchSubtree(0)
        # p1 = parents[1].searchSubtree(0)
        left_0 = parents[0].searchSubtree(1)
        left_1 = parents[1].searchSubtree(1)
        b0, e0 = self.searchSubtree_idx(parents[0], 1)
        b1, e1 = self.searchSubtree_idx(parents[1], 1)
        if e0 + 1 < len(parents[0]):
            right_0 = parents[0].searchSubtree(e0 + 1)
            flag0 = 1
        if e1 + 1 < len(parents[1]):
            right_1 = parents[1].searchSubtree(e1 + 1)
            flag1 = 1
        left = [left_0, left_1]
        if flag0 == 1 and flag1 == 1:
            right = [right_0, right_1]
            r_arity = 0
            if parents[0][e0 + 1].arity == parents[1][e1 + 1].arity:
                r_arity = 1
        # print(f"left: {left}")
        # print(f"right: {right}")
        r = random.randint(0, 1)  # r是root
        m = 1 - r
        if len(parents[r]) < len(parents[m]):
            # root = parents[r].root
            if flag1 == 0 or flag0 == 0:
                return parents[r], parents[m]
            parents[m][0] = parents[r].root
            m = r
        if flag0 == 1 and flag1 == 1:
            r1 = random.randint(0, 1)  # r1是左邊
            if parents[r][1] == parents[r1][1]:
                parents[r][left[r]] = parents[r1][left[r1]]
            if r_arity == 1:
                r2 = random.randint(0, 1)
                parents[r][right[r]] = parents[r2][right[r2]]
        else:
            r1 = random.randint(0, 1)
            parents[r][left[r1]] = parents[r1][left[r1]]

        return parents[r], parents[r]

    def cx_fair(self, ind1, ind2):
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

        while 1:
            index2 = random.choice(types2[type_])
            height2 = self.subtree_height(ind2, index2)
            if height2 <= height1:
                break
        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]
        return ind1, ind2

    def traverse_tree(self, stack, res, parent, idx):
        while res != 0:
            res -= 1
            idx += 1
            stack.append((parent[idx], [], idx))
            res += parent[idx].arity
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
        while idx1 < len(ind1) and idx2 < len(ind2):
            # Push the nodes to the stack
            stack1.append((ind1[idx1], [], idx1))
            stack2.append((ind2[idx2], [], idx2))

            # Not the same region
            if stack1[-1][0].arity != stack2[-1][0].arity:
                res1 = stack1[-1][0].arity
                res2 = stack2[-1][0].arity
                stack1, res1, idx1 = self.traverse_tree(stack1, res1, ind1, idx1)
                stack2, res2, idx2 = self.traverse_tree(stack2, res2, ind2, idx2)
            else:
                region1.append([ind1[idx1], idx1])
                region2.append([ind2[idx2], idx2])

            idx1 += 1
            idx2 += 1

        # for pri, idx in region1:
        #     print(f"{idx}: {pri.name}")

        # Select crossover point
        if len(region1) > 0:
            point = random.randint(0, len(region1) - 1)
            # print(f"crossover point: {point}")
            # print(f"crossover point for trees: {region1[point]}, {region2[point]}")

            # Swap subtrees
            slice1 = ind1.searchSubtree(region1[point][1])
            slice2 = ind2.searchSubtree(region2[point][1])
            ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

        return ind1, ind2

    def crossover(self, ind1_, ind2_):
        ind1 = copy.deepcopy(ind1_)
        ind2 = copy.deepcopy(ind2_)
        # No crossover on single node tree
        if len(ind1) < 2 or len(ind2) < 2:
            return ind1, ind2
        # Crossover
        if random.uniform(0, 1) < self.cx_pb:
            if self.cx_method == CX_RANDOM:
                choice = random.choice(
                    [
                        self.toolbox.cx_simple,
                        self.toolbox.cx_uniform,
                        self.toolbox.cx_fair,
                        self.toolbox.cx_one,
                    ]
                )
                try:
                    ind1, ind2 = choice(ind1, ind2)
                except:
                    pass
            if self.cx_method == CX_SIMPLE:
                try:
                    ind1, ind2 = self.toolbox.cx_simple(ind1, ind2)
                except:
                    pass
            if self.cx_method == CX_UNIFORM:
                try:
                    ind1, ind2 = self.toolbox.cx_uniform(ind1, ind2)
                except:
                    pass
            if self.cx_method == CX_FAIR:
                try:
                    ind1, ind2 = self.toolbox.cx_fair(ind1, ind2)
                except:
                    pass
            if self.cx_method == CX_ONE_POINT:
                try:
                    ind1, ind2 = self.toolbox.cx_one(ind1, ind2)
                except:
                    pass

        fitness_ind1 = self.toolbox.evaluate(ind1)
        fitness_ind2 = self.toolbox.evaluate(ind2)
        if fitness_ind1 <= fitness_ind2:
            return ind2
        else:
            return ind1

    def mutate(self, child):
        if random.random() < self.mut_pb:
            try:
                self.toolbox.mutate(child)
            except:
                pass
            child.fitness.values = self.toolbox.evaluate(child)
        return child

    def select(self):
        candidates = self.toolbox.select(self.pop)
        parents = candidates[0:3]
        sorted_parents = sorted(
            parents, key=lambda ind: ind.fitness.values
        )  # 小到大排序
        sorted_fitness = [ind.fitness.values for ind in sorted_parents]
        # print(f"sorted_fitness: {sorted_fitness}")
        offspring = self.crossover(sorted_parents[1], sorted_parents[2])
        offspring = self.mutate(offspring)
        off_fit = self.toolbox.evaluate(offspring)
        # print(f"offspring fitness: {off_fit}")
        if off_fit[0] >= sorted_fitness[0]:
            idx = self.pop.index(sorted_parents[0])
            # print(idx)
            self.pop[idx] = offspring
            # print(f"Ater selection: {self.pop[idx]}")
            # print(off_fit[0])
            self.pop[idx].fitness.values = self.toolbox.evaluate(offspring)

        return

    def write_record(self, writer):
        # print(f"Eval iteration: {self.eval_count}")
        record = self.stats.compile(self.pop)
        self.hof.update(self.pop)
        # print(record)
        # print(f"Best ind: {self.hof[0]}")
        best_ind = str(self.hof[0])
        row = [self.eval_count] + list(record.values()) + [best_ind]
        writer.writerow(row)

    def csv_name(self):
        return "*".join(
            [
                "-algo",
                self.algorithm,
                "-e",
                self.embedding_type,
                "-n",
                str(self.dim),
                "-p",
                str(self.pop_size),
                "-pc",
                str(self.cx_pb),
                "-pm",
                str(self.mut_pb),
                "-g",
                str(self.max_gen),
                "-c",
                str(self.cx_method),
                "-eval",
                str(self.max_eval),
                "-run",
                str(self.run),
            ]
        )

    def evolving(self):
        print("Start evolving...")
        # print(f"csv_:{self.csv_name()}")
        csv_name = self.csv_name()
        os.makedirs(f"{PATH}/results", exist_ok=True)
        # print(f"csv_name: {csv_name},run:{self.run}")

        with open(f"{PATH}/results/{csv_name}.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["eval_count", "avg", "std", "min", "max", "best_individual"]
            )
            while self.eval_count < self.max_eval:
                self.select()
                if self.eval_count % 1 == 0:
                    self.write_record(writer)
