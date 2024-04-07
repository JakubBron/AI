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

def get_element(populationFitness):
    i = random.uniform(0,1)
    for j, c in enumerate(populationFitness):
        if i<= c[0]:
            return c[1]

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
chance_margin = 0.8

for _ in range(generations):
    population_history.append(population)

    # TODO: implement genetic algorithm
    # Part one, selection (choose parents)
    totalFitness = 0
    populationFitness = []
    for p in population:
        fitness_ = fitness(items, knapsack_max_capacity, p)
        totalFitness = totalFitness + fitness_
        populationFitness.append([fitness_, p])

    for p in populationFitness:
        p[0] = p[0]/totalFitness

    populationFitness.sort()
    populationFitness.reverse()

    for i in range(1, len(populationFitness)):
        populationFitness[i][0] = populationFitness[i][0] + populationFitness[i-1][0]


    # Part two, crossover
    children = []   # array of bit words (stored as arrays of bits)
    for __ in range(int(population_size/2)):
        e1 = get_element(populationFitness)
        e2 = get_element(populationFitness)

        k11, k21 = e1[:int(len(e1)/2)], e2[int(len(e2)/2):]    
        k12, k22 = e2[:int(len(e2)/2)], e1[int(len(e2)/2):]

        k1 = k11+k22
        k2 = k12+k21
        children.append(k1)
        children.append(k2)

    # Part three, mutation
    for kid in children:
        index = random.randint(0, len(kid) - 1)
        kid[index] = not kid[index]


    # Part four, update population
    elite = populationFitness[:n_elite]
    population = []
    for e in elite:
        population.append(e[1])
    
    population = population + children[:(population_size-n_elite)]

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
