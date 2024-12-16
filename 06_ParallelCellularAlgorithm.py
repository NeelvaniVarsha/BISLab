import numpy as np
from multiprocessing import Pool

def fitness_function(x):
    return x**2 - 4*x + 4

def update_cell(cell_info):
    x, neighbors = cell_info
    new_value = np.mean(neighbors)  
    return new_value

def parallel_cellular_algorithm(grid_size, num_iterations):
    grid = np.random.uniform(-10, 10, (grid_size, grid_size))
    best_solution = None
    best_fitness = float('inf')

    def get_neighbors(i, j):
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = (i + di) % grid_size, (j + dj) % grid_size  
                neighbors.append(grid[ni, nj])
        return neighbors

    for iteration in range(num_iterations):
        inputs = []
        for i in range(grid_size):
            for j in range(grid_size):
                neighbors = get_neighbors(i, j)
                inputs.append((grid[i, j], neighbors))

        with Pool() as pool:
            updated_values = pool.map(update_cell, inputs)

        updated_grid = np.array(updated_values).reshape(grid_size, grid_size)
        grid = updated_grid

        for i in range(grid_size):
            for j in range(grid_size):
                fitness = fitness_function(grid[i, j])
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = grid[i, j]

        print(f"Iteration {iteration + 1}/{num_iterations}:")
        print(f"  Current Best Solution: {best_solution}")
        print(f"  Current Best Fitness: {best_fitness:.4f}")
        print(f"  Average Fitness of Grid: {np.mean([fitness_function(grid[i, j]) for i in range(grid_size) for j in range(grid_size)]):.4f}")
        print("-" * 50)

    return best_solution, best_fitness

if __name__ == "__main__":
    grid_size = int(input("Enter grid size: "))
    num_iterations = int(input("Enter number of iterations: "))
    
    print("\nStarting Parallel Cellular Algorithm...")
    print(f"Grid Size: {grid_size}x{grid_size}")
    print(f"Number of Iterations: {num_iterations}")
    print("-" * 50)
    
    solution, fitness = parallel_cellular_algorithm(grid_size, num_iterations)
    
    print("\nAlgorithm Completed!")
    print("=" * 50)
    print(f"Best Solution Found: {solution}")
    print(f"Fitness of Best Solution: {fitness:.4f}")
    print("=" * 50)
