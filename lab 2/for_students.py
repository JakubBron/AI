from itertools import compress
import random
import time
import matplotlib.pyplot as plt

from data import *

def initial_population(individual_size, population_size):
    # population = []
    # for i <= population_size: 
    #   individual = [] -> array of bits
    #   for j <= individual_size:
    #      individual.append(random.choice([True, False]))
    # plenty population, each has random assigment of items in the knapsack
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
#items, knapsack_max_capacity = get_small()
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

population_copy = population.copy()
chance_margin = 0.1

for _ in range(generations):
    population_history.append(population)

    # TODO: implement genetic algorithm
    # Part one, selection (choose parents)
    weights = []
    for individual in population:
        weights.append(fitness(items, knapsack_max_capacity, individual))
    
    # chocice includes weights
    parentsSelected = random.choices(population, weights=weights, k=n_selection)

    # Part two, crossover (create children)
    children = []   # array of bit words (stored as arrays of bits)
    half = len(items) // 2
    for parent1, parent2 in zip(parentsSelected[::2], parentsSelected[1::2]):
        child1 = parent1[:half] + parent2[half:]    # first half of parent1 and second half of parent2
        child2 = parent2[:half] + parent1[half:]    # first half of parent2 and second half of parent1
        children.append(child1)
        children.append(child2)

    # Part three, mutation (change randomly chosen childrens properties)
    """ 
        TO ASK:
        Every child should have a change??? We lose parents properties.
        Or some part of kids should have a mutation? We have a mix of parents propeties & mutated parents properties.
    """
    for child in children:
        chance = random.random()
        if chance < chance_margin:
            index = random.randint(0, len(child) - 1)
            child[index] = not child[index]

    # Part four, update population (new population = choosen survivors + children)
    eliteSurvivors = random.choices(population, k=population_size-len(children), weights=weights)
    population = eliteSurvivors + children

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
