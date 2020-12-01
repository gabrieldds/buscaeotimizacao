from utils import boundedUniformConvolution, generateRandomArray
import numpy as np
import utils
import random

class Individual:
    """
    docstring
    """
    def __init__(self, func_optmized, dim, min_v, max_v, os, fbias):
        self.values = np.zeros(dim)
        self.dim    = dim
        self.func_optmized = func_optmized
        self.min_v = min_v
        self.max_v = max_v
        self.os = os
        self.fbias = fbias
        self.fitness = 0.0

    def mutation(self, p=1.0, r=0.01):
        self.values = boundedUniformConvolution(solution=self.values, p=1.0, r=0.01, min=self.min_v, max=self.max_v)

    def generate(self):
        self.values = generateRandomArray(min=self.min_v, max=self.max_v, size=self.dim)

    def assess_fitness(self):
        self.fitness = self.func_optmized(self.values, self.os, self.fbias)
    
        