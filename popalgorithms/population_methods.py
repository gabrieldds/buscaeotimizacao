from copy import deepcopy
from popalgorithms.individualclass import Individual
from popalgorithms.populationclass import Population
from matplotlib import cm
from utils import boundedUniformConvolution
import numpy as np
import matplotlib.pyplot as plt
import cec
import utils
import random

class PopulationMethods:

    def __init__(self, func_optmized, os, min_v, max_v, dim, fbias, max_iterations):
        self.func_optmized  = func_optmized
        self.min_v = min_v
        self.max_v = max_v
        self.dim   = dim
        self.max_iterations = max_iterations
        self.fbias = fbias
        self.os = os
    
    def uv_evolution(self, u: int, v: int) -> Individual:
        population = Population(set())
        for _ in range(v):
            individual = Individual(func_optmized=self.func_optmized, dim=self.dim, min_v=self.min_v, 
                max_v=self.max_v, os=self.os, fbias=self.fbias)
            individual.generate()
            population.data.add(individual)

        best =  Individual(func_optmized=self.func_optmized, dim=self.dim, min_v=self.min_v, 
                    max_v=self.max_v, os=self.os, fbias=self.fbias)
        iteration = 0
        while iteration < self.max_iterations:
            for individual in population.data:
                individual.assess_fitness()
                if not best.values or individual.fitness < best.fitness:
                    best = deepcopy(individual)
        
            q = population.get_best_fitness(u)
            population = deepcopy(q)
            for individual in q.data:
                for _ in range(int(v / u)):
                    population.data.add(individual.mutation())
        
            iteration += 1
        
        return best.values

    def genetic_algorithm_elitism(self, pop_size: int, n: int) -> Individual:
        population = Population(set())
        for _ in range(pop_size):
            individual = Individual(func_optmized=self.func_optmized, dim=self.dim, min_v=self.min_v, 
                max_v=self.max_v, os=self.os, fbias=self.fbias)
            individual.generate()
            population.data.add(individual)
    
        best =  Individual(func_optmized=self.func_optmized, dim=self.dim, min_v=self.min_v, 
                max_v=self.max_v, os=self.os, fbias=self.fbias)
        iteration = 0
        while iteration < self.max_iterations:
            for individual in population.data:
                individual.assess_fitness()
                if np.array_equal(best.values, np.zeros(self.dim)) or individual.fitness < best.fitness:
                    best = deepcopy(individual)

            q = population.get_best_fitness(n)
            
            for _ in range(int((pop_size - n) / 2)):
                parentA = population.tournament_selection(2)
                parentB = population.tournament_selection(2)
                childrenA, childrenB = population.two_point_crossover(deepcopy(parentA), deepcopy(parentB))
                childrenA.assess_fitness()
                childrenB.assess_fitness()
                childrenA.mutation()
                childrenB.mutation()
                q.data.add(childrenA)
                q.data.add(childrenB)
            
            population = deepcopy(q)
        
            iteration += 1

        return best.values
    
    def differencial_evolution(self, gama: float, pop_size: int) -> Individual:
        population = Population(np.array([]))
        Q = Population(np.array([]))
        for _ in range(pop_size):
            individual = Individual(func_optmized=self.func_optmized, dim=self.dim, min_v=self.min_v, 
                max_v=self.max_v, os=self.os, fbias=self.fbias)
            individual.generate()
            population.data = np.append(population.data, individual)
        
        best = Individual(func_optmized=self.func_optmized, dim=self.dim, min_v=self.min_v, 
                max_v=self.max_v, os=self.os, fbias=self.fbias)
        iteration = 0
        while iteration < self.max_iterations:
            for i in range(len(population.data)):
                population.data[i].assess_fitness()
                if len(Q.data) != 0 and Q.data[i].fitness < population.data[i].fitness:
                    population.data[i] = deepcopy(Q.data[i])
                if np.array_equal(best.values, np.zeros(self.dim)) or population.data[i].fitness < best.fitness:
                    best = deepcopy(population.data[i])
                    #print(best.fitness)
            Q = deepcopy(population)
            for j in range(len(Q.data)):
                copy_q = deepcopy(Q)
                indexes = [index for index in range(len(Q.data)) if j != index]
                vector_a, vector_b, vector_c = copy_q.data[np.random.choice(indexes, 3, replace = False)]
                vector_d = Individual(func_optmized=self.func_optmized, dim=self.dim, min_v=self.min_v, 
                    max_v=self.max_v, os=self.os, fbias=self.fbias)
                vector_d.values = np.clip(vector_a.values + gama * (vector_b.values - vector_c.values),
                    self.min_v, self.max_v)
                child, _ = population.two_point_crossover(vector_d, deepcopy(Q.data[j]))
                population.data[j] = deepcopy(child)
            
            iteration += 1

        return best.values
        
    def exec(self, algo='', number_of_executions=10, u=2, v=4, pop_size = 8, n = 4, gama = 0.995):
        result = []
        if(algo == 'uv_evolution'):
            for _ in range(number_of_executions):
                solution = self.uv_evolution(u,v)
                best   = self.func_optmized(solution, self.os, self.fbias)
                result.append(best)
            return result
        elif algo == 'genetic_algorithm_elitism':
            for _ in range(number_of_executions):
                solution = self.genetic_algorithm_elitism(pop_size, n)
                best   = self.func_optmized(solution, self.os, self.fbias)
                result.append(best)
            return result
        elif algo == 'differencial_evolution':
            for _ in range(number_of_executions):
                solution = self.differencial_evolution(gama, pop_size)
                best   = self.func_optmized(solution, self.os, self.fbias)
                result.append(best)
            return result
        else:
            return []