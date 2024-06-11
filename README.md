# Evolutionary Computation Term Paper (Evolutionary Computation, 2023 spring)

**This READ.md is modified from (EC_hw_2024)[https://github.com/fffchameleon/EC_hw_2024/tree/main]. Thanks to the TAs for building the parser.py and its usage.

## Sample Code Usage
Clone this repository to your local machine,
```bash
git clone https://github.com/fffchameleon/EC_hw_2024.git
cd EC_hw_2024
```bash
cd py
python3 main.py -n 10 -r binary -p 100 -u 0 -c 0.9 -m 0.1 -g 500 -d
```
Both commands should output the following information,
```
-------------------------------------------
|Parameter           |Value               |
-------------------------------------------
|dimension           |10                  |
|representation      |binary              |
|population_size     |100                 |
|uniform_crossover   |false               |
|crossover_method    |2-point             |
|cross_prob          |0.9                 |
|mut_prob            |0.1                 |
|num_generations     |500                 |
-------------------------------------------
0.00145984
```
## Input/Output Format
We provide sample parser code for two languages (C++/Python). 

You can write your own parser, but it must be capable of accepting the following parameters, and your program must provide at least the following 8 options. 

p.s. You may add more options to make your experiments more convenient and complete.

| Options       | Description | Default |
| ------------- | ----------- | ------- |
| `-n, --dimension` | The dimension of Schwefel function | 10 |
| `-r, --representation`    | The representation to use. Binary or real-valued (binary, real) | binary |
| `-p, --population_size`	  |  Number of the population |100 |
| `-u, --uniform_crossover`  | The crossover method using uniform crossover (1) or not (0). If not, then for binary GA, it will use 2-point crossover and for real-valued GA will use whole arithmetic crossover | 0 |
| `-c, --pc` |	Probability for the crossover | $p_c$=0.9 |
| `-m, --pm` |	Probability for the mutation  |  $p_m$=0.1 |
| `-g, --generations`  |  Max number of generations to terminate | 500 |
| `-d, --debug`        | Turn on debug prints | false |
