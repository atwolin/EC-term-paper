# Evolutionary Computation Term Paper (Evolutionary Computation, 2023 spring)

**This READ.md is modified from [EC_hw_2024](https://github.com/fffchameleon/EC_hw_2024/tree/main). Thanks to the TAs for building the parser.py and its usage.**

## Sample Code Usage
Clone this repository to your local machine,
```bash
git clone https://github.com/atwolin/EC-term-paper.git
cd EC-term-paper
```

Run this command to exucute simple GP,
```bash
python main.py -algo "simple_gp" -e "word2vec" -n 10 -p 500 -pc 1.0 -pm 0.3 -c "cx_random" -eval 10000 -d
```
The command should output the following information,
```
-------------------------------------------
|Parameter           |Value               |
-------------------------------------------
|algorithm           |simple_gp           |
|embedding_type      |word2vec            |
|dimension           |10                  |
|population_size     |500                 |
|crossover_method    |cx_random           |
|cross_prob          |1.0                 |
|mut_prob            |0.3                 |
|num_generations     |500                 |
|num_evaluations     |10000               |
|num_runs            |1                   |
-------------------------------------------
```
and training the GP model.

## Input Format
We provide the following 8 Options to adjust.

| Options       | Description | Default | Choices |
| ------------- | ----------- | ------- | ------- |
| `-algo, --algorithm` | The algorithm to use. simple_gp, rf, or gpab | "simple_gp" | "simple_gp", "rf", "gpab" |
| `-e, --embedding_type` | The type of embedding model to use. word2vec, glove, or fasttext | "word2vec", "glove", "fasttext" |
| `-n, --dimension` | The dimension of word embeddings | 10 | |
| `-p, --population_size`	  | Number of the population |100 | |
| `-c --crossover_method` | The crossover method to use. cx_random, cx_simple, cx_uniform, cx_fair, or cx_one_point  | "cx_random" | "cx_random", "cx_simple", "cx_uniform", "cx_fair", "cx_one_point" |
| `-pc, --prob_crossover` |	Probability for the crossover | 1.0 | |
| `-pm, --prob_mutation` | Probability for the mutation | 0.1 | |
| `-g, --generations`  | Max number of generations to terminate | 500 | |
| `-eval evaluations` | Max number of evaluations to terminate | 1000 | |
| `-d, --debug`        | Turn on debug prints | false | |
