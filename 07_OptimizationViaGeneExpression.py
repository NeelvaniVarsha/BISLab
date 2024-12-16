import random

def fitness_function(x):
    return -1 * (x**2 - 4*x + 4)  

def decode(chromosome, lower_bound, upper_bound):
    max_value = 2**len(chromosome) - 1
    decoded = int(chromosome, 2)
    return lower_bound + (decoded / max_value) * (upper_bound - lower_bound)

def initialize_population(population_size, gene_length):
    return [''.join(random.choice('01') for _ in range(gene_length)) for _ in range(population_size)]

def evaluate_population(population, lower_bound, upper_bound):
    return [fitness_function(decode(ind, lower_bound, upper_bound)) for ind in population]

def select_parents(population, fitnesses):
    total_fitness = sum(fitnesses)
    selection_probs = [f / total_fitness for f in fitnesses]
    return random.choices(population, weights=selection_probs, k=2)

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

def mutate(chromosome, mutation_rate):
    return ''.join(bit if random.random() > mutation_rate else random.choice('01') for bit in chromosome)

def gene_expression_algorithm(population_size, gene_length, mutation_rate, crossover_rate, generations, lower_bound, upper_bound):
    population = initialize_population(population_size, gene_length)
    best_solution = None
    best_fitness = float('-inf')

    for generation in range(generations):
        fitnesses = evaluate_population(population, lower_bound, upper_bound)

        for ind, fit in zip(population, fitnesses):
            if fit > best_fitness:
                best_fitness = fit
                best_solution = ind

        print(f"Generation {generation + 1}/{generations}")
        print(f"  Best Solution So Far: {decode(best_solution, lower_bound, upper_bound):.4f}")
        print(f"  Best Fitness So Far: {best_fitness:.4f}")
        print(f"  Average Fitness: {sum(fitnesses) / len(fitnesses):.4f}")
        print("-" * 50)

        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitnesses)
            if random.random() < crossover_rate:
                offspring1, offspring2 = crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1, parent2
            new_population.append(mutate(offspring1, mutation_rate))
            new_population.append(mutate(offspring2, mutation_rate))
        
        population = new_population

    decoded_best = decode(best_solution, lower_bound, upper_bound)
    return decoded_best, best_fitness

if __name__ == "__main__":
    print("Gene Expression Algorithm for Optimization\n")
    population_size = int(input("Enter population size: "))
    gene_length = int(input("Enter gene length: "))
    mutation_rate = float(input("Enter mutation rate (e.g., 0.01): "))
    crossover_rate = float(input("Enter crossover rate (e.g., 0.7): "))
    generations = int(input("Enter number of generations: "))
    lower_bound = float(input("Enter lower bound of the solution space: "))
    upper_bound = float(input("Enter upper bound of the solution space: "))

    print("\nStarting Gene Expression Algorithm...")
    print("-" * 50)
    
    best_solution, best_fitness = gene_expression_algorithm(
        population_size, gene_length, mutation_rate, crossover_rate, generations, lower_bound, upper_bound
    )
    
    print("\nAlgorithm Completed!")
    print("=" * 50)
    print(f"Best Solution Found: {best_solution:.4f}")
    print(f"Fitness of Best Solution: {best_fitness:.4f}")
    print("=" * 50)
