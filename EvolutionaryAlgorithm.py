import time

from Parameter import *
from Constraint import *
from BitString import *


class EvolutionaryAlgorithm:

    def __init__(self, name, parameters, algorithm):
        self.name = name
        self.parameters = parameters
        self.algorithm = algorithm
        # List with all the evolutionary algorithms
        evolutionary_algorithms.append(self)
        # List with the names of the algorithms in order to display them in dropdown menus
        evolutionary_algorithm_names.append(self.name)

    # Solve a fitness function with the algorithm and returns the solution, running times and different iterations
    # get_path is set to True when the evolution of the algorithm needs to be displayed.
    # get_path is set to false in the script in order to use less memory
    def solve_fitness(self, evolutionary_parameters, size, fitness_function, fitness_parameters, get_path,
                      max_iter=float('inf'), get_proba=False):
        t1 = time.time()
        bit_string, iterations, x, y = self.algorithm(evolutionary_parameters,
                                                      size,
                                                      fitness_function,
                                                      fitness_parameters,
                                                      get_path,
                                                      max_iter,
                                                      get_proba)
        t2 = time.time()
        return bit_string.string, iterations, t2 - t1, x, y

    # Solve a SAT problem with the algorithm and returns the solution, running times and different iterations
    # get_path is set to True when the evolution of the algorithm needs to be displayed.
    # get_path is set to false in the script in order to use less memory
    def solve_SAT(self, evolutionary_parameters, sat_problem, get_path, max_iter=float('inf')):
        t1 = time.time()
        bit_string, iterations, x, y = self.algorithm(evolutionary_parameters,
                                                      sat_problem.number_of_variables,
                                                      sat_problem,
                                                      [],
                                                      get_path,
                                                      max_iter,
                                                      False)
        t2 = time.time()
        if iterations > max_iter:
            return bit_string, -sat_problem.result(False, bit_string), t2-t1, x, y
        else:
            return bit_string, iterations, t2 - t1,  x, y


# Algorithm for the (1+1) EA
def one_plus_one(parameters, n, fitness_function, fitness_parameters, get_path, max_iter, get_proba):
    r = parameters[0]
    # Creation of a random bit string of length n
    bit_string = BitString(n)
    # Compute the value of the fitness function for the previously created bit string
    fitness_value = fitness_function.result(fitness_parameters, bit_string)
    iterations = 1
    x = [1]
    y = [fitness_value]
    # Compute the maximum of the function in order to know when this value is reached
    fitness_maximum = fitness_function.maximum(fitness_parameters, n)
    # found_maximum is true if the maximum has been reached
    found_maximum = (fitness_value == fitness_maximum)
    while not found_maximum and iterations <= max_iter:
        # Creation of the offspring
        new_bit_string = bit_string.create_offspring_p(r/n)
        new_fitness_value = fitness_function.result(fitness_parameters, new_bit_string)
        iterations += 1
        # If the fitness value of the new bit string is better than before, it is kept
        if new_fitness_value >= fitness_value:
            bit_string = new_bit_string
            fitness_value = new_fitness_value
            found_maximum = (fitness_value == fitness_maximum)
            if get_path:
                x.append(iterations)
                y.append(fitness_value)
    return bit_string, iterations, x, y


# Algorithm for the SD-(1+1) EA
def sd_one_plus_one(parameters, n, fitness_function, fitness_parameters, get_path, max_iter, get_proba):
    R = parameters[0]
    # Creation of a random bit string of length "size"
    bit_string = BitString(n)
    # Compute the value of the fitness function for the previously created bit string
    fitness_value = fitness_function.result(fitness_parameters, bit_string)
    iterations = 1
    x = [1]
    y = [fitness_value]
    # Compute the maximum of the function in order to know when this value is reached
    fitness_maximum = fitness_function.maximum(fitness_parameters, n)
    # found_maximum is true if the maximum has been reached
    found_maximum = (fitness_value == fitness_maximum)
    # Number of iteration with the actual strength
    u = 0
    # Strength
    r = 1
    while not found_maximum and iterations <= max_iter:
        # Creation of the offspring
        new_bit_string = bit_string.create_offspring_p(r/n)
        new_fitness_value = fitness_function.result(fitness_parameters, new_bit_string)
        iterations += 1
        u += 1
        # If the fitness value of the new bit string is better than before, it is kept
        if new_fitness_value > fitness_value:
            bit_string = new_bit_string
            fitness_value = new_fitness_value
            found_maximum = (fitness_value == fitness_maximum)
            if get_path:
                x.append(iterations)
                y.append(fitness_value)
            # The strength is reset to 1 and the number of iterations at this strength to 0
            r = 1
            u = 0
        # If the fitness value is equal to the previous one and the strength is 1, the new bit string is taken
        elif new_fitness_value == fitness_value and r == 1:
            bit_string = new_bit_string
            fitness_value = new_fitness_value
            found_maximum = (fitness_value == fitness_maximum)
            if get_path:
                x.append(iterations)
                y.append(fitness_value)
        # After too many iterations, the strength is increased and the number of iterations reset to 0
        if u > np.power(n / r, r) * np.power(n / (n - r), n - r) * math.log(math.exp(1) * n * R):
            new_r = min(r+1, n/2)
            u = 0
        else:
            new_r = r
        r = new_r
    return bit_string, iterations, x, y


# Algorithm for the SASD-(1+lambda) EA
def sasd_one_plus_lambda(parameters, n, fitness_function, fitness_parameters, get_path, max_iter, get_proba):
    lbd = parameters[0]
    r_init = parameters[1]
    R = parameters[2]
    # Creation of a random bit string of length "size"
    bit_string = BitString(n)
    # Compute the value of the fitness function for the previously created bit string
    fitness_value = fitness_function.result(fitness_parameters, bit_string)
    iterations = 1
    x = [1]
    y = [fitness_value]
    # Compute the maximum of the function in order to know when this value is reached
    fitness_maximum = fitness_function.maximum(fitness_parameters, n)
    # found_maximum is true if the maximum has been reached
    found_maximum = (fitness_value == fitness_maximum)
    # Number of iteration with the actual strength
    u = 0
    # Strength
    r = r_init
    # Boolean variable indication stagnation detection
    g = False
    while not found_maximum and iterations <= max_iter:
        u = u + 1
        # Stagnation Detection
        if g:
            # Creation of lambda offsprings
            new_bit_strings = []
            new_fitness_values = []
            for i in range(lbd):
                new_bit_string = bit_string.create_offspring_p(r/n)
                new_bit_strings.append(new_bit_string)
                new_fitness_values.append(fitness_function.result(fitness_parameters, new_bit_string))
                iterations += 1
            # Picking the best offspring
            # In case of a tie, the first offspring in the array is returned
            # This is equivalent to breaking ties randomly as the order is random
            max_index = np.argmax(new_fitness_values)
            new_bit_string = new_bit_strings[max_index]
            new_fitness_value = new_fitness_values[max_index]
            # If the fitness value of the best new bit string is better than before, it is kept
            if new_fitness_value > fitness_value:
                bit_string = new_bit_string
                fitness_value = new_fitness_value
                found_maximum = (fitness_value == fitness_maximum)
                if get_path:
                    x.append(iterations)
                    y.append(fitness_value)
                # Reinitialise parameters
                new_r = r_init
                g = False
                u = 0
            else:
                if u > np.power(n / r, r) * np.power(n / (n - r), n - r) * math.log(math.exp(1) * n * R) / lbd:
                    new_r = min(r + 1, n/2)
                    u = 0
                else:
                    new_r = r
        # g = False
        # Self-Adjusting
        else:
            # Creation of lambda offsprings
            new_bit_strings = []
            new_fitness_values = []
            for i in range(lbd):
                if i <= lbd / 2 - 1:
                    p = r / (2 * n)
                else:
                    p = 2 * r / n
                new_bit_string = bit_string.create_offspring_p(p)
                new_bit_strings.append(new_bit_string)
                new_fitness_values.append(fitness_function.result(fitness_parameters, new_bit_string))
                iterations += 1
            # Picking the best offspring
            # In case of a tie, the first offspring in the array is returned
            # This is equivalent to breaking ties randomly as the order is random
            max_index = np.argmax(new_fitness_values)
            new_bit_string = new_bit_strings[max_index]
            new_fitness_value = new_fitness_values[max_index]
            if max_index <= lbd / 2 - 1:
                used_r = r / 2
            else:
                used_r = 2 * r
            # If the fitness value of the best new bit string is better than before, it is kept
            if new_fitness_value >= fitness_value:
                if new_fitness_value > fitness_value:
                    u = 0
                bit_string = new_bit_string
                fitness_value = new_fitness_value
                found_maximum = (fitness_value == fitness_maximum)
                if get_path:
                    x.append(iterations)
                    y.append(fitness_value)
            # Updating r
            if random.random() < 1/2:
                r = used_r
            else:
                if random.random() < 1/2:
                    r = r / 2
                else:
                    r = 2 * r
            new_r = min(max(2, r), n/4)
            # After too many iterations, the strength and iterations are reset
            # We go back to the Stagnation Detection
            if u > np.power(n / r, r) * np.power(n / (n - r), n - r) * math.log(math.exp(1) * n * R) / lbd:
                new_r = 2
                g = True
                u = 0
        r = new_r
    return bit_string, iterations, x, y


# Algorithm for the SD-RLS^r algorithm
def sd_rls_r(parameters, n, fitness_function, fitness_parameters, get_path, max_iter, get_proba):
    R = parameters[0]
    # Creation of a random bit-string of size n
    bit_string = BitString(n)
    fitness_value = fitness_function.result(fitness_parameters, bit_string)
    iterations = 1
    x = [1]
    y = [fitness_value]
    fitness_maximum = fitness_function.maximum(fitness_parameters, n)
    found_maximum = (fitness_value == fitness_maximum)
    r = 1
    s = 1
    u = 0
    while not found_maximum and iterations <= max_iter:
        new_bit_string = bit_string.create_offspring_s(s)
        new_fitness_value = fitness_function.result(fitness_parameters, new_bit_string)
        iterations += 1
        u += 1
        # Better bit-string found
        if new_fitness_value > fitness_value:
            bit_string = new_bit_string
            fitness_value = new_fitness_value
            found_maximum = (fitness_value == fitness_maximum)
            if get_path:
                x.append(iterations)
                y.append(fitness_value)
            # Resetting parameters
            r = 1
            s = 1
            u = 0
        # If same value and strength is still 1, update bit-string
        elif new_fitness_value == fitness_value and r == 1:
            bit_string = new_bit_string
            fitness_value = new_fitness_value
            found_maximum = (fitness_value == fitness_maximum)
            if get_path:
                x.append(iterations)
                y.append(fitness_value)
        # If too many iterations, update parameters
        if u > math.comb(n, s) * math.log(R):
            if s == 1:
                if r < n/2:
                    r += 1
                else:
                    # All the strengths are now checked if r >= n/2
                    r = n
                s = r
            else:
                s -= 1
            u = 0
    return bit_string, iterations, x, y


# Algorithm for the SD-RLS^m algorithm
def sd_rls_m(parameters, n, fitness_function, fitness_parameters, get_path, max_iter, get_proba):
    R = parameters[0]
    # Creation of a random bit-string of size n
    bit_string = BitString(n)
    fitness_value = fitness_function.result(fitness_parameters, bit_string)
    iterations = 1
    x = [1]
    y = [fitness_value]
    fitness_maximum = fitness_function.maximum(fitness_parameters, n)
    found_maximum = (fitness_value == fitness_maximum)
    r = 1
    s = 1
    u = 0
    B = float('inf')
    while not found_maximum and iterations <= max_iter:
        new_bit_string = bit_string.create_offspring_s(s)
        new_fitness_value = fitness_function.result(fitness_parameters, new_bit_string)
        iterations += 1
        u += 1
        # Better bit string found
        if new_fitness_value > fitness_value:
            bit_string = new_bit_string
            fitness_value = new_fitness_value
            found_maximum = (fitness_value == fitness_maximum)
            if get_path:
                x.append(iterations)
                y.append(fitness_value)
            # Memory of the strength of the update
            r = s
            s = 1
            if r > 1:
                B = u / (math.log(n) * (r-1))
            else:
                B = float('inf')
            u = 0
        # If same value and strength is still 1, update bit-string
        elif new_fitness_value == fitness_value and r == 1:
            bit_string = new_bit_string
            fitness_value = new_fitness_value
            found_maximum = (fitness_value == fitness_maximum)
            if get_path:
                x.append(iterations)
                y.append(fitness_value)
        # If too many iterations with a strength
        if u > min(B, math.comb(n, s) * math.log(R)):
            if s == r:
                if r < n/2:
                    r += 1
                else:
                    r = n
                s = 1
            else:
                s += 1
                if s == r:
                    B = float('inf')
            u = 0
    return bit_string, iterations, x, y


# Algorithm for the SA-(1, lambda) EA
def sa_one_lambda(parameters, n, fitness_function, fitness_parameters, get_path, max_iter, get_proba):
    lbd = parameters[0]
    F = parameters[1]
    r_init = parameters[2]
    # Creation of a random bit-string of size n
    bit_string = BitString(n)
    fitness_value = fitness_function.result(fitness_parameters, bit_string)
    # Storing the best element encountered in case max_iter is reached
    best_bit_string = bit_string
    best_fitness_value = fitness_value
    iterations = 1
    x = [1]
    y = [fitness_value]
    fitness_maximum = fitness_function.maximum(fitness_parameters, n)
    found_maximum = (fitness_value == fitness_maximum)
    r = r_init
    while not found_maximum and iterations <= max_iter:
        # Create offsprings with strength r/F or r*F
        rs = []
        new_bit_strings = []
        new_fitness_values = []
        for i in range(lbd):
            if random.random() < 1 / 2:
                rs.append(r/F)
            else:
                rs.append(F * r)
            new_bit_strings.append(bit_string.create_offspring_p(rs[i]/n))
            new_fitness_values.append(fitness_function.result(fitness_parameters, new_bit_strings[i]))
            iterations += 1
        index_max = 0
        # Finding the bit_string with the maximum fitness value favoring a mutation rate of r/F in case of tie
        # This is a random choice afterwards because the order is random
        for j in range(1, lbd):
            if (new_fitness_values[j] > new_fitness_values[index_max]) or \
                    (new_fitness_values[j] == new_fitness_values[index_max] and rs[j] == r/F):
                index_max = j
        # The best bit string is kept and will be used to create the future offsprings
        bit_string = new_bit_strings[index_max]
        fitness_value = new_fitness_values[index_max]
        found_maximum = (fitness_value == fitness_maximum)
        if fitness_value > best_fitness_value:
            best_bit_string = bit_string
            best_fitness_value = fitness_value
        new_r = rs[index_max]
        if get_path:
            x.append(iterations)
            y.append(fitness_value)
        # Bound the value of r
        r = min(max(F, new_r), n/(2 * F))
    return best_bit_string, iterations, x, y


# Algorithm for the (mu+1) EA with deterministic crowding
def mu_plus_one_deterministic(parameters, n, fitness_function, fitness_parameters, get_path, max_iter, get_proba):
    mu = parameters[0]
    population = []
    fitness_values = []
    iterations = 0
    # Creation of mu random bit strings
    for i in range(mu):
        # Add the random bit string to the population
        population.append(BitString(n))
        fitness_values.append(fitness_function.result(fitness_parameters, population[i]))
        iterations += 1
    # Storing the best bit string in order to know when the algorithm stops
    index_max = np.argmax(fitness_values)
    bit_string = population[index_max]
    fitness_value = fitness_values[index_max]
    x = [iterations]
    y = [fitness_value]
    fitness_maximum = fitness_function.maximum(fitness_parameters, n)
    found_maximum = (fitness_value == fitness_maximum)
    while not found_maximum and iterations <= max_iter:
        # Picking one of the bit strings randomly and create an offspring from it
        index = random.randrange(mu)
        new_bit_string = population[index].create_offspring_p(1 / n)
        new_fitness_value = fitness_function.result(fitness_parameters, new_bit_string)
        iterations += 1
        # If the offspring is better than the parent, the parent is replaced
        if new_fitness_value >= fitness_values[index]:
            population[index] = new_bit_string
            fitness_values[index] = new_fitness_value
            # If we found a new maximum, the maximum value of fitness is replaced
            if new_fitness_value >= fitness_value:
                bit_string = new_bit_string
                fitness_value = new_fitness_value
                found_maximum = (fitness_value == fitness_maximum)
                if get_path:
                    x.append(iterations)
                    y.append(fitness_value)
    return bit_string, iterations, x, y


# Algorithm for the (mu + 1) EA
# The parameter "operator" is used to pick a parent selection operator ("Uniform" or "Inverse-K")
def mu_plus_one(operator, parameters, n, fitness_function, fitness_parameters, get_path, max_iter, get_proba):
    mu = parameters[0]
    if operator == "Inverse-K":
        K = parameters[1]
    population = []
    fitness_values = []
    iterations = 0
    # Creating mu bit strings randomly
    for i in range(mu):
        population.append(BitString(n))
        fitness_values.append(fitness_function.result(fitness_parameters, population[i]))
        iterations += 1
    # Storing the best bit string in order to know when the algorithm stops
    fitness_maximum = fitness_function.maximum(fitness_parameters, n)
    index_max = np.argmax(fitness_values)
    bit_string = population[index_max]
    fitness_value = fitness_values[index_max]
    x = [iterations]
    y = [fitness_value]
    found_maximum = (fitness_value == fitness_maximum)
    while not found_maximum and iterations <= max_iter:
        # Pick an element depending on the parent selection operator
        # Uniformly at random
        if operator == "Uniform":
            # Pick a random element
            index = random.randrange(mu)
        # Inverse K-tournament
        elif operator == "Inverse-K":
            # Pick K random element and select the worst one
            indexes = random.sample(list(range(0, mu)), K)
            index = indexes[0]
            fitness_min = fitness_values[index]
            for idx in indexes:
                if fitness_values[idx] < fitness_min:
                    index = idx
                    fitness_min = fitness_values[idx]
        else:
            raise Exception("Operator not defined")
        # Create an offspring using the selected bit string
        new_bit_string = population[index].create_offspring_p(1 / n)
        new_fitness_value = fitness_function.result(fitness_parameters, new_bit_string)
        iterations += 1
        min_indexes = [0]
        fitness_min = fitness_values[0]
        # Get the worst bit string of the mu bit strings (breaking ties randomly) in order to replace it if better found
        for i in range(1, mu):
            if fitness_values[i] == fitness_min:
                min_indexes.append(i)
            elif fitness_values[i] < fitness_min:
                min_indexes = [i]
                fitness_min = fitness_values[i]
        new_index = random.choice(min_indexes)
        # If the new bit string is an improvement, replace the worst one by it
        if new_fitness_value >= fitness_values[new_index]:
            population[new_index] = new_bit_string
            fitness_values[new_index] = new_fitness_value
            if new_fitness_value >= fitness_value:
                bit_string = new_bit_string
                fitness_value = new_fitness_value
                found_maximum = (fitness_value == fitness_maximum)
                if get_path:
                    x.append(iterations)
                    y.append(fitness_value)
    return bit_string, iterations, x, y


# Algorithm for the compact Genetic Algorithm
def compact_genetic_algorithm(parameters, n, fitness_function, fitness_parameters, get_path, max_iter, get_proba):
    K = parameters[0]
    # Initialise the probabilities of each bit to 1/2
    ps = np.full(n, 1/2)
    fitness_maximum = fitness_function.maximum(fitness_parameters, n)
    found_maximum = False
    iterations = 0
    xs = []
    ys = []
    while not found_maximum and iterations <= max_iter:
        # Generate 2 bit strings randomly using the probabilities
        x = BitString(n)
        y = BitString(n)
        x_string = ""
        y_string = ""
        for i in range(n):
            if random.random() < ps[i]:
                x_string += "1"
            else:
                x_string += "0"
            if random.random() < ps[i]:
                y_string += "1"
            else:
                y_string += "0"
        x.string = x_string
        y.string = y_string
        x_fitness = fitness_function.result(fitness_parameters, x)
        y_fitness = fitness_function.result(fitness_parameters, y)
        iterations += 2
        # Switch strings in order to have x as the best one
        if x_fitness < y_fitness:
            x, y = y, x
            x_fitness, y_fitness = y_fitness, x_fitness
        if get_path:
            xs.append(iterations)
            ys.append(x_fitness)
        found_maximum = (x_fitness == fitness_maximum)
        # Weight probabilities depending on the results
        for i in range(n):
            if x.string[i] > y.string[i]:
                ps[i] += 1/K
            elif x.string[i] < y.string[i]:
                ps[i] -= 1/K
            ps[i] = max(min(ps[i], 1 - 1/n), 1/n)
    if not get_path and get_proba:
        xs = list(range(1, n+1))
        ys = ps
    return x, iterations, xs, ys


# List containing every evolutionary algorithm
evolutionary_algorithms = []
evolutionary_algorithm_names = []

# Creation of the (1+1) EA
# Strength -> Every bit if flipped with a probability of strength/size
Strength = Parameter("Strength", "integer", 1, 1, "size/2", INT)
OnePlusOne = EvolutionaryAlgorithm("(1+1) EA", [Strength], one_plus_one)

# Creation of the SD-(1+1) EA
# R -> it is used to control the probability of failing to find an improvement at the "right" strength,
paramR = Parameter("R", "integer", "size^3", "size^3", float('inf'), INT)
SDOnePlusOne = EvolutionaryAlgorithm("SD-(1+1) EA", [paramR], sd_one_plus_one)

# Creation of the SASD-(1+lambda) EA
# Lambda -> Number of offsprings created from the parent
Lambda = Parameter("Lambda", "integer", 10, 2, float('inf'), M2)
Initial_Strength = Parameter("Initial strength", "integer", 1, 1, "size/2", INT)
paramR = Parameter("R", "integer", "size^3", "size^3", float('inf'), INT)
SASDOnePlusLambda = EvolutionaryAlgorithm("SASD-(1+lambda) EA",
                                          [Lambda, Initial_Strength, paramR],
                                          sasd_one_plus_lambda)

# Creation of the SD-RLS_r
paramR = Parameter("R", "integer", "size^3.1",  "size^3.1", float('inf'), INT)
SDRLSR = EvolutionaryAlgorithm("SD-RLS^r", [paramR], sd_rls_r)

# Creation of the SD-RLS_m
paramR = Parameter("R", "integer", "size^3.1", "size^3.1", float('inf'), INT)
SDRLSM = EvolutionaryAlgorithm("SD-RLS^m", [paramR], sd_rls_m)

# Creation of the self-adjusting (1, lambda) EA
Lambda = Parameter("Lambda", "integer", "size", "ln(size)", float('inf'), INT)
paramF = Parameter("F", "integer", 2, 2, float('inf'), INT)
Initial_Strength = Parameter("Initial strength", "integer", 2, 2, "size", INT)
SAOneLambda = EvolutionaryAlgorithm("SA-(1, lambda) EA", [Lambda, paramF, Initial_Strength], sa_one_lambda)

# Creation of the (mu + 1) EA with deterministic crowding
Mu = Parameter("mu", "integer", "size^1.2", 1, float('inf'), INT)
MuPlusOneDeterministic = EvolutionaryAlgorithm("(mu+1) EA D", [Mu], mu_plus_one_deterministic)

# Creation of the (mu + 1) EA with uniform selection
Mu = Parameter("mu", "integer", "size^1.2", 1, float('inf'), INT)
MuPlusOneUniform = EvolutionaryAlgorithm("(mu+1) EA U", [Mu], functools.partial(mu_plus_one, "Uniform"))

# Creation of the (mu + 1) EA with Inverse K-tournament selection
Mu = Parameter("mu", "integer", "size^1.2", 1, float('inf'), INT)
K = Parameter("K", "integer", 5, 1, float('inf'), INT)
MuPlusOneInverseK = EvolutionaryAlgorithm("(mu+1) EA I",
                                          [Mu, K],
                                          functools.partial(mu_plus_one, "Inverse-K"))

# Creation of the compact Genetic Algorithm
K = Parameter("K", "integer", "sqrt*ln", 1, float('inf'), INT)
cGA = EvolutionaryAlgorithm("cGA", [K], compact_genetic_algorithm)
