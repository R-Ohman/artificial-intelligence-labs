from itertools import compress
import random
import time
import matplotlib.pyplot as plt

from data import *


def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]


def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))


def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness


items, knapsack_max_capacity = get_big()
print(items)

population_size = 100
generations = 200
n_selection = 20
n_elite = 1

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)
for _ in range(generations):
    population_history.append(population)

    # 1. Roulette wheel selection
    total_fitness = sum([fitness(items, knapsack_max_capacity, individual) for individual in population])
    probability_distribution = [fitness(items, knapsack_max_capacity, individual) / total_fitness for individual in population]
    selected = random.choices(population, weights=probability_distribution, k=n_selection)

    # 2. Crossover
    new_population = []
    for _ in range(population_size - n_elite):
        parent1, parent2 = random.choices(selected, k=2)
        crossover_point = random.randint(1, len(items) - 1)
        children = parent1[:crossover_point] + parent2[crossover_point:]
        new_population.append(children)

    # 3. Mutation
    for i in range(len(new_population)):
        for j in range(len(new_population[i])):
            if random.random() < 0.01:
                new_population[i][j] = not new_population[i][j]

    # 4. Actualize with elites
    elite_population = sorted(population, key=lambda x: fitness(items, knapsack_max_capacity, x))[-n_elite:]
    new_population.extend(elite_population)
    population = new_population

    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
