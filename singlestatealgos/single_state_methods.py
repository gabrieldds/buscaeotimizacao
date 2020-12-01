from utils import generateRandomArray
import numpy as np
import random

class SingleStateMethods:
    def __init__(self, max_iterations, tweak, quality, min_v, max_v, dim, os, fbias):
        self.max_iterations = max_iterations
        self.tweak = tweak
        self.quality = quality
        self.min_v = min_v
        self.max_v = max_v
        self.dim = dim
        self.os = os
        self.fbias = fbias
    
    def hill_climbing(self, p=1.0, r=0.125) -> np.ndarray:
        S = generateRandomArray(self.min_v, self.max_v, self.dim)
        iteration = 0
        while iteration < self.max_iterations:
            R = self.tweak(solution=np.copy(S), min=self.min_v, max=self.max_v, p=p, r=r)
            if self.quality(x=R, Os=self.os, fbias=self.fbias) < self.quality(x=S, Os=self.os, fbias=self.fbias):
                S = np.copy(R)
            iteration += 1
        return S
    
    def simulated_annealing(self, p=1.0, r=0.125, t=1000, alpha=0.995) -> np.ndarray:
        S = generateRandomArray(self.min_v, self.max_v, self.dim)
        best = np.copy(S)
        iteration = 0
        while iteration < self.max_iterations or t >= 0.0001:
            R = self.tweak(solution=np.copy(S), min=self.min_v, max=self.max_v, p=p, r=r)
            quality_R = self.quality(x=R, Os=self.os, fbias=self.fbias)
            quality_S = self.quality(x=S, Os=self.os, fbias=self.fbias)
            delta     = (quality_R - quality_S)
            expo = np.exp(-(delta) / t)
            if (delta <= 0 or random.uniform(0, 1) < expo):
                S = np.copy(R)
            t = t * alpha
            if self.quality(x=S, Os=self.os, fbias=self.fbias) < self.quality(x=best, Os=self.os, fbias=self.fbias):
                best = np.copy(S)
            iteration += 1

        return best

    def solution(self, s):
        return self.quality(s, self.os, self.fbias)

    def exec(self, algo='', number_of_executions=10, p=1.0, r=0.125, t=1000, alpha=0.995):
        result = []
        if(algo == 'hill_climbing'):
            for _ in range(number_of_executions):
                solution = self.hill_climbing(p, r)
                best   = self.quality(solution, self.os, self.fbias)
                result.append(best)
            return result
        elif algo == 'simulated_annealing':
            for _ in range(number_of_executions):
                solution = self.simulated_annealing(p, r, t, alpha)
                best   = self.quality(solution, self.os, self.fbias)
                result.append(best)
            return result
        else:
            return []



