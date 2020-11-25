from popalgorithms import PopulationMethods
from cec import Functions
from utils import boundedUniformConvolution, get_optimal_vector, write_result
from random import uniform, randint
import numpy as np

os = get_optimal_vector('otimo-rastringin.txt')
functions = Functions()

population_methods_rastringin = PopulationMethods(func_optmized=functions.rastrigin, 
        min_v=-5, max_v=5, dim=100, fbias=-330.0, max_iterations=3000, os=os)

values = []
gamas = []
pop_sizes = []
for _ in range(10):
    gama = uniform(0.5, 1.0)
    pop_size = randint(4, 21)
    solution_de_rastrigin = population_methods_rastringin.differencial_evolution(gama, pop_size)
    values.append(functions.rastrigin(solution_de_rastrigin, os, -330))
    gamas.append(gama)
    pop_sizes.append(pop_size)

for i in range(10):
    print('{} {} {}'.format(gamas[i], pop_sizes[i], values[i]))

