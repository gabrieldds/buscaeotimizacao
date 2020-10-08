import numpy as np
from numpy import random as randPy
import random

def sphereFunction(x, Os, fbias):
    """
    docstring
    """
    sum = np.sum((x - Os)**2)
    return sum + fbias

def generateRandomArray(min, max, size):
    return np.array([random.uniform(min, max) for _ in range(size)])

def quality(solution, Os, target):
    return sphereFunction(solution, Os, target)

def boundedUniformConvolution(solution, p=1.0, r=0.1, min=-100, max=100):
    n = 0.0
    for i in range(len(solution)):
        if p >= generateRandomArray(0, 1, 1):
            while True:
                n = generateRandomArray(-r, r, 1)
                if min <= solution[i] + n <= max:
                    break
            solution[i] += n
    return solution

def myTweak(solution):
    for i in range(len(solution)):
        solution[i] += generateRandomArray(-0.1, 0.1, 1)
    return solution

def hillClimbing(tweakFunction, nIterations, Os, fbias):
    best = generateRandomArray(-100, 100, 100)
    iteration = 0
    while True:
        if(round(quality(best, Os, fbias)) == round(fbias) or iteration >= nIterations):
            break
        possibleSolution = tweakFunction(np.copy(best))
        if(quality(possibleSolution, Os, fbias) < quality(best, Os, fbias)):
            best = np.copy(possibleSolution)
        iteration += 1
    return best

Os = np.loadtxt("otimo-f1.txt")
#s = hillClimbing(boundedUniformConvolution, 50000, Os, -450.0)
#print(sphereFunction(s, Os, -450.0))
'''
Execução com algoritmo 8
algoritmo8 = open("hillClimbingAlgo8.txt", "w")
txt = ""
for exec in range(10):
    s = hillClimbing(boundedUniformConvolution, 50000, Os, -450.0)
    result_sphere = sphereFunction(s, Os, -450.0)
    txt = "iteracao: {}\narray_solucao: {}\nsphere_result: {}\n".format(exec, s, result_sphere)
    algoritmo8.write(txt)

algoritmo8.close()
'''

# Execução com meu tweak
myTweakFile = open("hillClimbingMyTweak.txt", "w")
txtMyTweak = ""
for exec in range(10):
    s = hillClimbing(myTweak, 50000, Os, -450.0)
    result_sphere = sphereFunction(s, Os, -450.0)
    txtMyTweak = "iteracao: {}\narray_solucao: {}\nsphere_result: {}\n".format(exec, s, result_sphere)
    myTweakFile.write(txtMyTweak)

myTweakFile.close()