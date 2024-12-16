import numpy as np

class AntColony:
    def __init__(self, distance_matrix, num_ants, num_iterations, decay):
        self.distance_matrix = distance_matrix
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.decay = decay
        self.num_cities = len(distance_matrix)
        self.pheromone = np.ones((self.num_cities, self.num_cities))

    def run(self):
        best_path = None
        best_distance = float('inf')

        for _ in range(self.num_iterations):
            all_paths = self.generate_all_paths()
            for path, distance in all_paths:
                if distance < best_distance:
                    best_path = path
                    best_distance = distance
            self.update_pheromones(all_paths)

            self.pheromone *= (1 - self.decay)

        return best_path, best_distance

    def generate_all_paths(self):
        all_paths = []
        for _ in range(self.num_ants):
            path = self.generate_path()
            distance = self.calculate_total_distance(path)
            all_paths.append((path, distance))
        return all_paths

    def generate_path(self):
        path = [0]
        visited = {0}

        while len(path) < self.num_cities:
            current_city = path[-1]
            next_city = self.choose_next_city(current_city, visited)
            path.append(next_city)
            visited.add(next_city)

        path.append(0)
        return path

    def choose_next_city(self, current_city, visited):
        probabilities = []
        for city in range(self.num_cities):
            if city not in visited:
                probabilities.append(self.pheromone[current_city][city] / self.distance_matrix[current_city][city])
            else:
                probabilities.append(0)

        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum() if probabilities.sum() > 0 else 1

        return np.random.choice(range(self.num_cities), p=probabilities)

    def calculate_total_distance(self, path):
        return sum(self.distance_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1))

    def update_pheromones(self, all_paths):
        for path, distance in all_paths:
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i + 1]] += 1.0 / distance

if __name__ == "__main__":
    num_cities = int(input("Enter the number of cities: "))
    print("Enter the distance matrix row by row, separated by spaces (use 0 for the diagonal):")

    distance_matrix = []
    for i in range(num_cities):
        row = list(map(float, input(f"Row {i + 1}: ").strip().split()))
        if len(row) != num_cities:
            print("Error: Each row must have the same number of elements as the number of cities.")
            exit(1)
        distance_matrix.append(row)

    distance_matrix = np.array(distance_matrix)

    num_ants = int(input("Enter the number of ants: "))
    num_iterations = int(input("Enter the number of iterations: "))
    decay = float(input("Enter the pheromone decay rate (e.g., 0.1): "))

    aco = AntColony(distance_matrix, num_ants, num_iterations, decay)

    best_path, best_distance = aco.run()

    print("Best path: ", best_path)
    print("Best distance: ", best_distance)
