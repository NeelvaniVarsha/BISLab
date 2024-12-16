import random

POPULATION_SIZE = int(input("Enter Population size: "))
GENE_LENGTH = int(input("Enter Gene length: "))
MUTATION_RATE = float(input("Enter Mutation rate: "))
GENERATIONS = int(input("Enter Generations: "))

def fitness(x):
    return x**2

def decode(binary_str):
    return int(binary_str, 2)

def create_population():
    return [''.join(random.choice('01') for _ in range(GENE_LENGTH)) for _ in range(POPULATION_SIZE)]

def evaluate_population(population):
    return [fitness(decode(individual)) for individual in population]

def select_parents(population, fitnesses):
    total_fitness = sum(fitnesses)
    selection_probs = [f / total_fitness for f in fitnesses]
    return random.choices(population, weights=selection_probs, k=2)

def crossover(parent1, parent2):
    point = random.randint(1, GENE_LENGTH - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

def mutate(individual):
    return ''.join(bit if random.random() > MUTATION_RATE else random.choice('01') for bit in individual)

def genetic_algorithm():
    population = create_population()

    for generation in range(GENERATIONS):
        fitnesses = evaluate_population(population)
        new_population = []

        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = select_parents(population, fitnesses)
            offspring1, offspring2 = crossover(parent1, parent2)
            new_population.append(mutate(offspring1))
            new_population.append(mutate(offspring2))

        population = new_population

    best_individual = max(population, key=lambda ind: fitness(decode(ind)))
    best_fitness = fitness(decode(best_individual))

    return decode(best_individual), best_fitness

if __name__ == "__main__":
    best_solution, best_fitness = genetic_algorithm()
    print(f"Best solution: {best_solution}, Fitness: {best_fitness}")
