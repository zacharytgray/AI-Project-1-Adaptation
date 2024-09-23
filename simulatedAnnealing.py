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

def simulatedAnnealing(problem, schedule, x0, stoppingIter, rangeMin, rangeMax, bitLength=52):
    """
    Simulated Annealing algorithm for function optimization.

    :param problem: A tuple (f, f_c) where f is the function to be minimized and f_c is the constraint function.
    :param schedule: A function that returns the temperature at time t.
    :param x0: The initial solution.
    :param stoppingIter: The maximum number of iterations.
    :param rangeMin: The minimum value of the range.
    :param rangeMax: The maximum value of the range.
    :param bitLength: The length of the bitstring representation.
    :return: The best solution found and its value.
    """
    f, fC = problem
    current = x0
    currentValue = f(current)
    bestSolution = current
    bestValue = currentValue

    for t in range(1, stoppingIter + 1):
        T = schedule(t)
        if T == 0:
            return bestSolution, bestValue
        
        # Select a random neighbor
        nextSolution = current + np.random.uniform(-1, 1, size=current.shape)
        
        # Ensure the neighbor satisfies the constraints
        if not fC(nextSolution):
            continue
        
        # Clip the neighbor values within the specified range
        nextSolution = np.clip(nextSolution, rangeMin, rangeMax)
        
        # Convert neighbor to bitstring and back to real values
        nextSolutionBitstrings = [realToBitstring(val, rangeMin, rangeMax, bitLength) for val in nextSolution]
        nextSolutionReals = np.array([bitstringToReal(bs, rangeMin, rangeMax) for bs in nextSolutionBitstrings])
        
        # Calculate the value of the neighbor
        nextValue = f(nextSolutionReals)
        
        # Calculate the change in value
        deltaE = currentValue - nextValue # If positive, the neighbor is better.
        
        # Move to the neighbor with a probability based on the acceptance probability
        if deltaE > 0 or random.random() < np.exp(deltaE / T):
            current = nextSolutionReals
            currentValue = nextValue
            
            # Update the best solution found
            if currentValue < bestValue: # less than because we are minimizing
                bestSolution = current
                bestValue = currentValue

    return bestSolution, bestValue

# Exponential Schedule function
def exponentialSchedule(t, T0=1000, alpha=0.96):
    return T0 * (alpha ** t)

# Logarithmic Schedule function
def logarithmicSchedule(t, T0=1000, c=1):
    return T0 / (1 + c * np.log(1 + t))

# Define the ranges for each problem
problemRanges = {
    'sphere': (-5, 5),
    'griew': (0, 200),
    'shekel': (0, 10),
    'micha': (-100, 100),
    'langermann': (0, 10),
    'odd_square': (-5 * np.pi, 5 * np.pi),
    'bump': (0, 100),
    'rastrigin': (-5.12, 5.12),
    'ackley': (-32.768, 32.768),
}

if __name__ == '__main__':
    stoppingIter = 1000

    # Define the problems
    problems = [
        (gae.sphere, gae.sphere_c, 'sphere'),
        (gae.griew, gae.griew_c, 'griew'),
        (gae.shekel, gae.shekel_c, 'shekel'),
        (gae.micha, gae.micha_c, 'micha'),
        (gae.langermann, gae.langermann_c, 'langermann'),
        (gae.odd_square, gae.odd_square_c, 'odd_square'),
        (gae.bump, gae.bump_c, 'bump'),
        (gae.rastrigin, gae.rastrigin_c, 'rastrigin'),
        (gae.ackley, gae.ackley_c, 'ackley'),
    ]

    for f, fC, name in problems: # f is the function, fC is the constraint function, name is the problem name
        rangeMin, rangeMax = problemRanges[name]
        x0 = np.random.uniform(rangeMin, rangeMax, size=(2,)) # Initial solution within the specified range
        bestSolution, bestValue = simulatedAnnealing((f, fC), logarithmicSchedule, x0, stoppingIter, rangeMin, rangeMax) 
        print(f"Best solution for {f.__name__}: {bestSolution}")
        print(f"Best value for {f.__name__}: {bestValue}\n")