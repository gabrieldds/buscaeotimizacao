import numpy as np

class Functions:
    def sphereFunction(self, x=np.array([]), Os=np.array([]), fbias=0.0):
        sum = 0.0
        if x.shape == Os.shape:
            sum = np.sum((x - Os)**2)
        return sum + fbias

    def rosenbrockFunction(self, x=np.array(np.array(np.array([]))), Os=np.array([]), fbias=0.0):
        sum = 0.0
        if x.shape[0] == Os.shape[0]:
            z = x - Os + 1
            for i in range(len(Os) - 1):
                sum += (100 * (z[i]**2 - z[i+1])**2 + (z[i] - 1)**2)
        return sum + fbias

    def rastrigin(self, x=np.array([]), Os=np.array([]), fbias=0.0):
        sum = 0.0
        if x.shape[0] == Os.shape[0]:
            z = x - Os
            sum = np.sum((z ** 2) - 10 * np.cos(2 * np.pi * z) + 10)
        return sum + fbias

    def ackley(self, x=np.array([]), Os=np.array([]), fbias=0.0):
        sum = 0.0
        if x.shape[0] == Os.shape[0]:
            z = x - Os
            D = len(z)
            sum = (-20 * np.exp(-0.2 * np.sqrt((1 / D) * np.sum(z ** 2))) 
               - np.exp((1 / D) * np.sum(np.cos(2 * np.pi * z))) + 20 + np.exp(1))
        return sum + fbias
    
        
