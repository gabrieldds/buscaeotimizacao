from popalgorithms import Population, Individual, PopulationMethods
from singlestatealgos import SingleStateMethods
from cec import Functions
from utils import boundedUniformConvolution, get_optimal_vector, write_result
import numpy as np

def main():
    os = get_optimal_vector('otimo-rastringin.txt')
    os_ackley = get_optimal_vector('otimo-ackley.txt')
    functions = Functions()
    '''single_state_methods = SingleStateMethods(max_iterations=50000, tweak=boundedUniformConvolution,
        quality=functions.rastrigin, min_v=-5, max_v=5, dim=100, os=os, fbias=-330.0)
    
    ### rastringin
    result_hill_climbing = open('result_rastringin_hill_climbing.txt', 'w')
    solution_hill = single_state_methods.exec('hill_climbing', p=0.0015259714771550525, r=2.1509364573536605)
    write_result(result_hill_climbing, solution_hill)
    
    result_simulated_annealing = open('result_rastringin_simulated_annealing.txt', 'w')
    solution_annealing = single_state_methods.exec('simulated_annealing', p=0.05141042211662372, r=4.745608374465435, 
        t=9394.0, alpha=0.7361746102651252)
    write_result(result_simulated_annealing, solution_annealing)
    
    ### ackley
    single_state_methods_ackley = SingleStateMethods(max_iterations=50000, tweak=boundedUniformConvolution,
        quality=functions.ackley, min_v=-32, max_v=32, dim=100, os=os_ackley, fbias=-140.0)
    
    result_hill_climbing_ackley = open('result_ackley_hill_climbing.txt', 'w')
    solution_hill_ackley = single_state_methods_ackley.exec('hill_climbing', p=0.004839690329197288, r=-6.065832947580056)
    write_result(result_hill_climbing_ackley, solution_hill_ackley)

    result_simulated_annealing_ackley = open('result_ackley_simulated_annealing.txt', 'w')
    solution_annealing_ackley = single_state_methods_ackley.exec('simulated_annealing', p=0.016289502897416153, r=15.977678504359758, 
        t=8137.0, alpha=0.8196299043624595)
    write_result(result_simulated_annealing_ackley, solution_annealing_ackley)
    
    ### population methods
    '''
    population_methods_rastringin = PopulationMethods(func_optmized=functions.rastrigin, 
        min_v=-5, max_v=5, dim=100, fbias=-330.0, max_iterations=50000, os=os)

    result_genetic_algorithm_rastrigin = open('result_genetic_algorithm_rastrigin.txt', 'w')
    solution_genetic_algo_rastrigin    = population_methods_rastringin.exec('genetic_algorithm_elitism', 10,
        pop_size=23, n=21, p=0.43963244924982525, r=1.2894809801906142)
    write_result(result_genetic_algorithm_rastrigin, solution_genetic_algo_rastrigin)

    #result_de_rastrigin = open('resul_de_rastrigin.txt', 'w')
    #solution_de_rastrigin = population_methods_rastringin.exec('differencial_evolution')
    #write_result(result_de_rastrigin, solution_de_rastrigin)

    population_methods_ackley = PopulationMethods(func_optmized=functions.ackley, 
        min_v=-32, max_v=32, dim=100, fbias=-140.0, max_iterations=50000, os=os_ackley)
    
    result_genetic_algorithm_ackley = open('result_genetic_algorithm_ackley.txt', 'w')
    solution_genetic_algo_ackley    = population_methods_ackley.exec('genetic_algorithm_elitism', 10,
        pop_size=21, n=15, p=0.2867959978583713, r=-16.487300571276784)
    write_result(result_genetic_algorithm_ackley, solution_genetic_algo_ackley)

    #result_de_ackley = open('resul_de_ackley.txt', 'w')
    #solution_de_ackley = population_methods_ackley.exec('differencial_evolution')
    #write_result(result_de_ackley, solution_de_ackley)


if __name__ == "__main__":
    main()