import numpy as np
def rastrigin_function(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))
class GreyWolfOptimizer:
    def __init__(self, cost_function, dim, pop_size, max_iter, lb, ub):
        self.cost_function = cost_function
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.alpha_position = np.zeros(dim)
        self.beta_position = np.zeros(dim)
        self.delta_position = np.zeros(dim)
        self.alpha_score = float('inf')
        self.beta_score = float('inf')
        self.delta_score = float('inf')
        self.positions = np.random.uniform(lb, ub, (pop_size, dim))
    def optimize(self):
        for t in range(self.max_iter):
            for i in range(self.pop_size):
                fitness = self.cost_function(self.positions[i])
                if fitness < self.alpha_score:
                    self.alpha_score = fitness
                    self.alpha_position = self.positions[i]
                elif fitness < self.beta_score:
                    self.beta_score = fitness
                    self.beta_position = self.positions[i]
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_position = self.positions[i]
            A1 = 2 * np.random.random(self.dim) - 1
            A2 = 2 * np.random.random(self.dim) - 1
            A3 = 2 * np.random.random(self.dim) - 1
            C1 = 2 * np.random.random(self.dim)
            C2 = 2 * np.random.random(self.dim)
            C3 = 2 * np.random.random(self.dim)
            for i in range(self.pop_size):
                D_alpha = np.abs(C1 * self.alpha_position - self.positions[i])
                D_beta = np.abs(C2 * self.beta_position - self.positions[i])
                D_delta = np.abs(C3 * self.delta_position - self.positions[i])
                self.positions[i] = self.positions[i] + A1 * D_alpha + A2 * D_beta + A3 * D_delta
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
            if (t + 1) % 10 == 0:
                print(f"Iter = {t+1} best fitness = {self.alpha_score:.3f}")
        return self.alpha_position, self.alpha_score
if __name__ == "__main__":
    dim = 3
    pop_size = 50
    max_iter = 100
    lb = -5.12
    ub = 5.12
    gwo = GreyWolfOptimizer(cost_function=rastrigin_function, dim=dim, pop_size=pop_size, max_iter=max_iter, lb=lb, ub=ub)
    best_position, best_score = gwo.optimize()
    print("\nGWO completed")
    print(f"\nBest solution found: {best_position}")
    print(f"Fitness of best solution = {best_score:.6f}")
    print("\nEnd GWO for Rastrigin function")
