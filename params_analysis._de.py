from popalgorithms import PopulationMethods
from singlestatealgos import SingleStateMethods
from cec import Functions
from utils import boundedUniformConvolution, get_optimal_vector, write_result
from random import uniform, randint
import numpy as np

os_rastringin = get_optimal_vector('otimo-rastringin.txt')
os_ackley = get_optimal_vector('otimo-ackley.txt')
functions = Functions()

def random_hill_parameters(search_method, min_f, max_f, max_exec=20):
    p_best = uniform(0.0, 1.0)
    r_best = uniform(min_f, max_f)
    best   = search_method.solution(search_method.hill_climbing(p_best, r_best))
    results = [{}]
    iteration = 0
    while iteration <= max_exec:
        p_s = uniform(0.0, 1.0)
        r_s = uniform(min_f, max_f)
        s   = search_method.solution(search_method.hill_climbing(p_s, r_s))
        if s < best:
            p_best = p_s
            r_best = r_s
            best   = s
        result = {}
        result['p'] = p_best
        result['r'] = r_best
        result['s'] = best
        results.append(result)
        iteration += 1

    return results


def random_simulated_annealing_parameters(search_method, min_f, max_f, max_exec=20):
    p_best = uniform(0.0, 1.0)
    r_best = uniform(min_f, max_f)
    t_best = float(randint(1000, 10000))
    alpha_best  = uniform(0.5, 0.999)
    best   = search_method.solution(search_method.simulated_annealing(p_best, r_best, t_best, alpha_best))
    results = [{}]
    iteration = 0
    while iteration <= max_exec:
        p_s = uniform(0.0, 1.0)
        r_s = uniform(min_f, max_f)
        t_s = float(randint(1000, 10000))
        alpha_s  = uniform(0.5, 0.999)
        s   = search_method.solution(search_method.simulated_annealing(p_s, r_s, t_s, alpha_s))
        if s < best:
            p_best = p_s
            r_best = r_s
            t_best = t_s
            alpha_best = alpha_s
            best   = s
        result = {}
        result['p'] = p_best
        result['r'] = r_best
        result['t'] = t_best
        result['alpha'] = alpha_best
        result['s'] = best
        results.append(result)
        iteration += 1

    return results

def random_ga_parameters(search_method, min_f, max_f, max_exec=20):
    p_best = uniform(0.0, 1.0)
    r_best = uniform(0.0, max_f)
    pop_size_best = randint(4, 30)
    n_best = randint(2, pop_size_best)
    if (pop_size_best - n_best) % 2 != 0:
        pop_size_best += 1
    best   = search_method.solution(search_method.genetic_algorithm_elitism(pop_size_best, n_best, p_best, r_best))
    results = [{}]
    iteration = 0
    while iteration <= max_exec:
        p_s = uniform(0.0, 1.0)
        r_s = uniform(0.0, max_f)
        pop_size_s = randint(4, 30)
        n_s = randint(2, pop_size_s)
        if (pop_size_s - n_s) % 2 != 0:
            pop_size_s += 1
        s   = search_method.solution(search_method.genetic_algorithm_elitism(pop_size_s, n_s, p_s, r_s))
        if s < best:
            p_best = p_s
            r_best = r_s
            pop_size_best = pop_size_s
            n_best = n_s
            best   = s
        result = {}
        result['p'] = p_best
        result['r'] = r_best
        result['pop_size'] = pop_size_best
        result['n'] = n_best
        result['s'] = best
        results.append(result)
        iteration += 1

    return results

def random_de_parameters(search_method, min_f, max_f, max_exec=20):
    p_best = uniform(0.0, 1.0)
    r_best = uniform(0.0, max_f)
    pop_size_best = randint(4, 30)
    gama_best = uniform(0.5, 1.0)
    best   = search_method.solution(search_method.differencial_evolution(gama_best, pop_size_best, p_best, r_best))
    results = [{}]
    iteration = 0
    while iteration <= max_exec:
        p_s = uniform(0.0, 1.0)
        r_s = uniform(0.0, max_f)
        pop_size_s = randint(4, 30)
        gama_s = uniform(0.5, 1.0)
        s   = search_method.solution(search_method.differencial_evolution(gama_s, pop_size_s, p_s, r_s))
        if s < best:
            p_best = p_s
            r_best = r_s
            pop_size_best = pop_size_s
            gama_best = gama_s
            best   = s
        result = {}
        result['p'] = p_best
        result['r'] = r_best
        result['pop_size'] = pop_size_best
        result['gama'] = gama_best
        result['s'] = best
        results.append(result)
        iteration += 1

    return results

single_state_methods_rastrigin = SingleStateMethods(max_iterations=25000, tweak=boundedUniformConvolution,
        quality=functions.rastrigin, min_v=-5, max_v=5, dim=100, os=os_rastringin, fbias=-330.0)

single_state_methods_ackley = SingleStateMethods(max_iterations=25000, tweak=boundedUniformConvolution,
        quality=functions.ackley, min_v=-32, max_v=32, dim=100, os=os_ackley, fbias=-140.0)

population_methods_rastringin = PopulationMethods(func_optmized=functions.rastrigin, 
        min_v=-5, max_v=5, dim=100, fbias=-330.0, max_iterations=25000, os=os_rastringin)

population_methods_ackley = PopulationMethods(func_optmized=functions.ackley, 
        min_v=-32, max_v=32, dim=100, fbias=-140.0, max_iterations=25000, os=os_ackley)


# hill_rastrigin_params = open('hill_rastrigin_params.txt', 'w')
# write_result(hill_rastrigin_params, random_hill_parameters(single_state_methods_rastrigin, -5, 5))

#simulated_rastrigin_params = open('simulated_rastrigin_params.txt', 'w')
#write_result(simulated_rastrigin_params, random_simulated_annealing_parameters(single_state_methods_rastrigin, -5, 5))

# hill_ackley_params = open('hill_ackley_params.txt', 'w')
# write_result(hill_ackley_params, random_hill_parameters(single_state_methods_ackley, -32, 32))

# simulated_ackley_params = open('simulated_ackley_params.txt', 'w')
# write_result(simulated_ackley_params, random_simulated_annealing_parameters(single_state_methods_ackley, -32, 32))

#ga_rastrigin_params = open('ga_rastrigin_params.txt', 'w')
#ga_ackley_params = open('ga_ackley_params.txt', 'w')
de_rastrigin_params = open('de_rastrigin_params.txt', 'w')
de_ackley_params = open('de_ackley_params.txt', 'w')
#write_result(ga_rastrigin_params, random_ga_parameters(population_methods_rastringin, -5, 5))
#write_result(ga_ackley_params, random_ga_parameters(population_methods_ackley, -32, 32))
write_result(de_rastrigin_params, random_de_parameters(population_methods_rastringin, -5, 5))
write_result(de_ackley_params, random_de_parameters(population_methods_ackley, -32, 32))