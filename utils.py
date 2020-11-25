import random
import numpy as np
import cec
from copy import deepcopy

def generateRandomArray(min, max, size):
    if(size == 1):
        return random.uniform(min, max)
    return np.array([random.uniform(min, max) for _ in range(size)])

def boundedUniformConvolution(solution, p=1.0, r=0.125, min=-5, max=5):
    n = 0.0
    for i in range(len(solution)):
        if p >= generateRandomArray(0, 1, 1):
            while True:
                n = generateRandomArray(-r, r, 1)
                if min <= solution[i] + n and solution[i] + n <= max:
                    break
            solution[i] += n
    return solution

def get_optimal_vector(file=''):
    os = np.loadtxt(file)
    return os

def write_result(writer, result):
    for i in range(len(result)):
        txt = "iteration: {}\nresult: {}\n".format(i, result[i])
        writer.write(txt)
    writer.close()
