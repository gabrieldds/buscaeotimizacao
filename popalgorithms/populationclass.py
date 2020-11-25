from popalgorithms.individualclass import Individual
from copy import deepcopy
import numpy as np
import operator
import random

class Population:
    """
    docstring
    """
    def __init__(self, data):
        self.data = data
    
    def get_best_fitness(self, number_of_bests: int):
        sorted_population = sorted(self.data, key=operator.attrgetter('fitness'))
        bests = []
        for i in range(number_of_bests):
            bests.append(sorted_population[i])
        return Population(set(bests))

    def tournament_selection(self, t: int) -> Individual:
        data = list(self.data)
        best = deepcopy(data[np.random.randint(len(data))])
        for _ in range(2, t):
            next = deepcopy(data[np.random.randint(len(data))])
            if next.fitness() < best.fitness():
                best = deepcopy(next)
        return best
    
    def two_point_crossover(self, v: Individual, w: Individual) -> tuple:
        c = random.randint(0, len(v.values))
        d = random.randint(0, len(w.values))

        if c > d:
            c, d = d, c
    
        if c != d:
            for i in range(c, d, 1):
                v.values[i], w.values[i] = w.values[i], v.values[i]
    
        return v, w