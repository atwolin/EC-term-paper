import os
import re
import gp
from simple_gp import simple_gp
from data import get_testing_embeddings
from sklearn.metrics.pairwise import cosine_similarity

cur_path = os.getcwd()
PATH = re.search(r"(.*EC-term-paper)", cur_path).group(0)
# print(PATH)

def test():
    data, embeddings, embedding_model = get_testing_embeddings("fasttext", 10)
    print(data)
    best_records = []
    for filename in os.listdir('best_ind_records'):
        # 构建文件的完整路径
        file_path = os.path.join('best_ind_records', filename)

        # 检查路径是否为文件
        if os.path.isfile(file_path):
            # 打开文件并进行处理（例如读取内容）
            with open(file_path, 'r') as file:
                lines = file.readlines()
                if len(lines) >= 2:
                    best_individual_line = lines[1]
                    # 提取最佳个体部分
                    best_individual = best_individual_line.split("Best individual: ")[1].strip()
        # 将文件名和最佳个体添加到记录中
        best_records.append((filename, best_individual))

    gp_instance = gp.GP(
        algorithm='simple_gp',
        embedding_type='fasttext',
        dimension=15,
        population_size=3,
        crossover_method=1,
        cross_prob=0.1,
        mut_prob=0.5,
        num_generations=500,
        num_evaluations=10,
        data=data,
        embeddings=embeddings,
        run=3
    )
    gp_instance.register()

    for line in best_records:
        print(line[1])
        x = gp_instance.eval(line[1], data, embedding_model)
        print(x) 
    #print(best_records)

# def eval(self, expression, data, model):
#         individual = gp.PrimitiveTree.from_string(expression, self.pset)
#         inputword_ = data.str.split(" ").apply(lambda x: x[:5])
#         realword_ = data.str.split(" ").str.get(5)
#         func = gp.compile(individual, self.pset)
#         similarity_=[]
#         for data_index in range(len(inputword_)):
#             words = inputword_.iloc[data_index]
#             in_vectors = [self.embeddings[word] for word in words]
#             a, b, c, d, e = in_vectors[:5]
#             y = realword_.iloc[data_index]
#             out_vector = self.embeddings[y]
#             predict = self.clean_data(func(a, b, c, d, e))
#             similarity_.append(cosine_similarity([predict], [out_vector])[0][0])
#             if self.embedding_type == "word2vec":
#                 outword = model.wv.most_similar(positive=[predict], topn=1)
#                 print(f"預測結果：{outword}")
#         # elif self.embeddings_model == "glove":
#         #     outword = model.wv.most_similar(positive=[predict], topn=1)
#         # elif self.embeddings_model == "fasttext":
#         #     outword = model.get_nearest_neighbors(predict, k=1)
#         # print(f"預測結果：{outword}")


if __name__ == "__main__":
    test()



            