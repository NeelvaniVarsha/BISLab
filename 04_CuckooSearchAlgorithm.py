import numpy as np
import math  

def objective_function(x):
    return np.sum(x**2)

def levy_flight(Lambda):
    sigma = (math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
             (math.gamma((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2)))**(1 / Lambda)
    u = np.random.normal(0, sigma, size=dimension)
    v = np.random.normal(0, 1, size=dimension)
    step = u / np.abs(v)**(1 / Lambda)
    return step

def cuckoo_search(n=5, max_iter=5, lb=-5, ub=5, pa=0.25, Lambda=1.5):
    nests = np.random.uniform(lb, ub, (n, dimension))
    fitness = np.array([objective_function(nest) for nest in nests])
    best_nest = nests[np.argmin(fitness)]
    best_fitness = np.min(fitness)

    for iteration in range(max_iter):
        for i in range(n):
            step_size = levy_flight(Lambda)
            new_nest = nests[i] + step_size
            new_nest = np.clip(new_nest, lb, ub)
            new_fitness = objective_function(new_nest)

            if new_fitness < fitness[i]:
                nests[i] = new_nest
                fitness[i] = new_fitness

        for i in range(n):
            if np.random.rand() < pa:
                nests[i] = np.random.uniform(lb, ub, dimension)
                fitness[i] = objective_function(nests[i])

        current_best_fitness = np.min(fitness)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_nest = nests[np.argmin(fitness)]

        print(f"Iteration {iteration + 1}/{max_iter}, Best Fitness: {best_fitness}")

    print("\nBest Solution Found:")
    print(f"Solution: {best_nest}")
    print(f"Fitness: {best_fitness}")

dimension = int(input("Number of dimensions (variables): "))       
nests = int(input("Number of nests (population size): "))            
max_iterations = int(input("Maximum number of iterations: "))   
lb = int(input("Lower bound for the solution space: "))              
ub = int(input("Upper bound for the solution space: "))              
pa = float(input("Probability of discovering an egg: "))            

cuckoo_search(n=nests, max_iter=max_iterations, lb=lb, ub=ub, pa=pa)
