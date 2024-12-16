import numpy as np

def objective_function(x):
    return -x**2 + 5*x + 20

def initialize_particles(n_particles, bounds):
    positions = np.random.uniform(bounds[0], bounds[1], n_particles)
    velocities = np.zeros(n_particles)
    return positions, velocities

def pso(n_particles, bounds, max_iter, w, c1, c2):
    positions, velocities = initialize_particles(n_particles, bounds)
    pbest_positions = np.copy(positions)
    pbest_values = objective_function(pbest_positions)
    gbest_position = pbest_positions[np.argmax(pbest_values)]
    gbest_value = max(pbest_values)

    for iteration in range(max_iter):
        print(f"Iteration {iteration + 1}:")

        for i in range(n_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest_positions[i] - positions[i]) +
                             c2 * r2 * (gbest_position - positions[i]))
            positions[i] += velocities[i]

            fitness_value = objective_function(positions[i])
            print(f"Particle {i+1}: Position = {positions[i]:.4f}, Velocity = {velocities[i]:.4f}, Fitness = {fitness_value:.4f}")

            if fitness_value > pbest_values[i]:
                pbest_positions[i] = positions[i]
                pbest_values[i] = fitness_value

        if max(pbest_values) > gbest_value:
            gbest_position = pbest_positions[np.argmax(pbest_values)]
            gbest_value = max(pbest_values)

        print(f"Global Best Position = {gbest_position:.4f}, Global Best Fitness = {gbest_value:.4f}\n")

    return gbest_position, gbest_value

if __name__ == "__main__":
    n_particles = int(input("Enter the number of particles: "))
    bounds = (-10, 10)
    max_iter = int(input("Enter the number of iterations: "))
    w = float(input("Enter the inertia weight (e.g., 0.5): "))
    c1 = float(input("Enter the cognitive constant (e.g., 1.5): "))
    c2 = float(input("Enter the social constant (e.g., 1.5): "))

    best_position, best_value = pso(n_particles, bounds, max_iter, w, c1, c2)
    print(f"Optimal Solution: Position = {best_position:.4f}, Value = {best_value:.4f}")
