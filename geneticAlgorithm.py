import numpy as np
import random
import ga_eval as gae
import ga_util as gau

def bitstringToReal(bitstring, rangeMin, rangeMax):
    # Convert a bitstring to a real number within the specified range
    return gau.bitstr2float(bitstring) * (rangeMax - rangeMin) + rangeMin

def realToBitstring(real, rangeMin, rangeMax, bitLength=52):
    # Convert a real number to a bitstring representation
    normalized = (real - rangeMin) / (rangeMax - rangeMin)
    bitstring = bin(int(normalized * (2**bitLength)))[2:].zfill(bitLength)
    return bitstring

def geneticAlgorithm(population, fitness, mutationProbability, maxGenerations):

    # param population: Initial population of individuals (bitstrings).
    # param fitness: Fitness function to evaluate individuals.
    # param mutationProbability: Probability of mutation.
    # param maxGenerations: Maximum number of generations.
    # return: The best individual found and its fitness value.\

    def weightedRandomChoices(population, weights, k=2):
        totalWeight = sum(weights)
        normalizedWeights = [w / totalWeight for w in weights]
        return random.choices(population, normalizedWeights, k=k)

    def doSinglePointCrossover(parent1, parent2): # Single-point crossover operator
        # pick a point to split the bitstrings, then combine them
        n = len(parent1)
        c = random.randint(1, n - 1)
        return parent1[:c] + parent2[c:]
    
    def doUniformCrossover(parent1, parent2): # Uniform crossover operator
        child = ''.join(random.choice([p1, p2]) for p1, p2 in zip(parent1, parent2))
        return child

    def mutate(individual):
        mutationIndices = random.sample(range(len(individual)), 3)
        mutated = list(individual)
        for index in mutationIndices: # Flip 3 bits at the selected indices
            mutated[index] = '1' if mutated[index] == '0' else '0'
        return ''.join(mutated)

    for _ in range(maxGenerations):
        weights = [1 / (fitness(individual) + 1e-6) for individual in population]  # Inverse of fitness for minimization
        population2 = []

        for _ in range(len(population)):
            parent1, parent2 = weightedRandomChoices(population, weights)
            child = doSinglePointCrossover(parent1, parent2)
            # child = doUniformCrossover(parent1, parent2)
            if random.random() < mutationProbability:
                child = mutate(child)
            population2.append(child)

        population = population2

        # Check for termination condition
        bestIndividual = min(population, key=fitness)
        if fitness(bestIndividual) < 1e-6:  # Example fitness threshold
            break

    bestIndividual = min(population, key=fitness)
    return bestIndividual, fitness(bestIndividual)

if __name__ == '__main__':
    populationSize = 100
    dimensions = 2
    bitLength = 52
    maxGenerations = 1000
    mutationProbability = 0.02

    # Define the problems
    problems = [
        (gae.sphere, gae.sphere_c, 'sphere', (-5, 5)),
        (gae.griew, gae.griew_c, 'griew', (0, 200)),
        (gae.shekel, gae.shekel_c, 'shekel', (0, 10)),
        (gae.micha, gae.micha_c, 'micha', (-100, 100)),
        (gae.langermann, gae.langermann_c, 'langermann', (0, 10)),
        (gae.odd_square, gae.odd_square_c, 'odd_square', (-5 * np.pi, 5 * np.pi)),
        (gae.bump, gae.bump_c, 'bump', (0, 100)),
        (gae.rastrigin, gae.rastrigin_c, 'rastrigin', (-5.12, 5.12)),
        (gae.ackley, gae.ackley_c, 'ackley', (-32.768, 32.768)),
    ]

    for f, fC, name, (rangeMin, rangeMax) in problems:
        # Initialize population with bitstrings
        population = [''.join(random.choice('01') for _ in range(bitLength * dimensions)) for _ in range(populationSize)]
        
        def fitness(bitstring):
            realValues = np.array([bitstringToReal(bitstring[i*bitLength:(i+1)*bitLength], rangeMin, rangeMax) for i in range(dimensions)])
            return f(realValues) if fC(realValues) else 1e6  # Penalize individuals that do not satisfy the constraints with a large finite value

        bestBitstring, bestFitness = geneticAlgorithm(population, fitness, mutationProbability=mutationProbability, maxGenerations=maxGenerations)
        bestIndividual = np.array([bitstringToReal(bestBitstring[i*bitLength:(i+1)*bitLength], rangeMin, rangeMax) for i in range(dimensions)])
        
        # Calculate the actual objective function value for the best individual
        actualBestFitness = f(bestIndividual)
        
        print(f"Best solution for {name}: {bestIndividual}")
        print(f"Best fitness for {name}: {actualBestFitness}\n")