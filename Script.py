import math
import os.path
from Interface import *


def general_run(evolutionary_algorithm,
                fitness_function,
                start_length, end_length, length_step, runs, nb_cores):

    # Create array with all the length analysed
    lengths = []
    for iteration in range(start_length, end_length + 1, length_step):
        lengths.append(iteration)

    # Initialisation
    results = []
    header = ""
    print(evolutionary_algorithm.name)
    print(fitness_function.name)

    # Running the algorithm
    for j in range(len(lengths)):
        # Header for the results in order to know the algorithm used and the problem size
        header += evolutionary_algorithm.name + " " + str(lengths[j]) + ";"
        # Printing the parameters of the algorithm for each problem size
        print("Length: " + str(lengths[j]) + ";" + "Parameters: "
              + str(default_parameters(evolutionary_algorithm, lengths[j])))
        # Append the "runs" runtimes on OneMax
        results.append(run_parallel(runs, lengths[j],
                                    evolutionary_algorithm,
                                    default_parameters(evolutionary_algorithm, lengths[j]),
                                    fitness_function,
                                    default_parameters(fitness_function, lengths[j]),
                                    nb_cores))

    print(header)
    for k in range(runs):
        line = ""
        for j in range(len(lengths)):
            line += str(results[j][k]) + ";"
        print(line)


def cga_prob(evolutionary_algorithm, fitness_function, length, runs, max_iter, nb_cores):

    print(evolutionary_algorithm.name)
    print(fitness_function.name)
    print("Length: " + str(length) + ";" + "Parameters: " + str(default_parameters(evolutionary_algorithm, length)))
    yss = run_parallel_cga(runs, length, max_iter,
                           evolutionary_algorithm,
                           default_parameters(evolutionary_algorithm, length),
                           fitness_function,
                           default_parameters(fitness_function, length),
                           nb_cores)

    header = "iterations;"
    for i in range(length):
        header += str(i+1) + ";"
    print(header)
    for i in range(runs):
        line = ""
        for j in range(len(yss[i])):
            line += str(yss[i][j]) + ";"
        print(line)


# Evaluate all the algorithms on OneMax
def script_1_1(start_length, end_length, length_step, runs, nb_cores):
    MuPlusOneDeterministic.parameters[0].default_value = 10
    MuPlusOneInverseK.parameters[0].default_value = 10
    MuPlusOneUniform.parameters[0].default_value = 10
    for i in range(len(evolutionary_algorithms)):
        general_run(evolutionary_algorithms[i],
                    OneMax,
                    start_length, end_length, length_step, runs, nb_cores)


# Evaluate cGA on OneMax with different values for K (coef * sqrt(n) * ln(n))
def script_1_2(start_length, end_length, length_step, runs, nb_cores):
    coefs = ["sqrt*ln", "5sqrt*ln", "25sqrt*ln", "125sqrt*ln", "625sqrt*ln", "3125sqrt*ln"]

    for i in range(len(coefs)):
        cGA.parameters[0].default_value = coefs[i]
        general_run(cGA, OneMax, start_length, end_length, length_step, runs, nb_cores)


# Evaluate cGA on OneMax with different values for K (coef * ln(n))
def script_1_3(start_length, end_length, length_step, runs, nb_cores):
    coefs = ["ln", "5ln", "25ln", "125ln", "625ln", "3125ln"]
    for i in range(len(coefs)):
        cGA.parameters[0].default_value = coefs[i]
        general_run(cGA, OneMax, start_length, end_length, length_step, runs, nb_cores)


# Evaluate the selected algorithms on Jump_4
def script_2_1(index, start_length, end_length, length_step, runs, nb_cores):
    algorithms = [(OnePlusOne, [4]),
                  (SDOnePlusOne, ["size^3"]),
                  (SAOneLambda, [10, 2, 2]),
                  (SDRLSR, ["size^3"])]

    for i in range(len(algorithms[index][1])):
        algorithms[index][0].parameters[i].default_value = algorithms[index][1][i]

    general_run(algorithms[index][0], JumpM,
                start_length, end_length, length_step, runs, nb_cores)


# Evaluate the cGA on the selected fitness function
def script_2_2(index_algorithm, index_fitness, start_length, end_length, length_step, runs, nb_cores):
    algorithms = [(cGA, ["sqrt*ln"])]
    functions = [(JumpOffsetM, [4]),
                 (JumpOffsetSpikeM, [4]),
                 (JumpOffsetSpikeM, [8])]

    for i in range(len(algorithms[index_algorithm][1])):
        algorithms[index_algorithm][0].parameters[i].default_value = algorithms[index_algorithm][1][i]

    for i in range(len(functions[index_fitness][1])):
        functions[index_fitness][0].parameters[i].default_value = functions[index_fitness][1][i]

    general_run(algorithms[index_algorithm][0], functions[index_fitness][0],
                start_length, end_length, length_step, runs, nb_cores)


# Evaluate the cGA with the selected parameter K on the selected fitness function
def script_2_3(index_algorithm, index_fitness, start_length, end_length, length_step, runs, max_iter, nb_cores):
    algorithms = [(cGA, ["sqrt*ln"]),
                  (cGA, ["5sqrt*ln"]),
                  (cGA, ["25sqrt*ln"]),
                  (cGA, ["125sqrt*ln"])]
    functions = [(JumpM, [4]),
                 (JumpOffsetM, [4]),
                 (JumpOffsetSpikeM, [4]),
                 (JumpOffsetSpikeM, [8])]

    for i in range(len(algorithms[index_algorithm][1])):
        algorithms[index_algorithm][0].parameters[i].default_value = algorithms[index_algorithm][1][i]

    for i in range(len(functions[index_fitness][1])):
        functions[index_fitness][0].parameters[i].default_value = functions[index_fitness][1][i]

    for length in range(start_length, end_length+1, length_step):
        cga_prob(algorithms[index_algorithm][0], functions[index_fitness][0],
                 length, runs, max_iter, nb_cores)


# Evalate the selected algorithm on the selected fitness function
def script_3_1(index_algorithm, index_fitness, start_length, end_length, length_step, runs, nb_cores):
    algorithms = [(SDOnePlusOne, ["size^3"]),
                  (SASDOnePlusLambda, [10, 1, "size^3"]),
                  (SDRLSR, ["size^3.1"]),
                  (SDRLSM, ["size^3.1"]),
                  (SAOneLambda, ["size", 2, 2]),
                  (MuPlusOneInverseK, [10, 5]),
                  (cGA, ["sqrt*ln"]),
                  (OnePlusOne, [1]),
                  (OnePlusOne, [4]),
                  (OnePlusOne, [2])]
    functions = [(JumpM, [4]),
                 (JumpOffsetM, [4]),
                 (JumpOffsetSpikeM, [8]),
                 (CliffD, [4]),
                 (HurdleW, [4]),
                 (NeedGlobalMutM, [])]

    for i in range(len(algorithms[index_algorithm][1])):
        algorithms[index_algorithm][0].parameters[i].default_value = algorithms[index_algorithm][1][i]

    for i in range(len(functions[index_fitness][1])):
        functions[index_fitness][0].parameters[i].default_value = functions[index_fitness][1][i]

    general_run(algorithms[index_algorithm][0], functions[index_fitness][0],
                start_length, end_length, length_step, runs, nb_cores)


# Evaluate the selected algorithm on the selected folder of SAT instances
def script_4_1(index_algorithm, folder_name, runs, max_iter, nb_cores):
    algorithms = [
        (OnePlusOne, [1]),
        (SDOnePlusOne, ["size^3"]),
        (SASDOnePlusLambda, [10, 1, "size^3"]),
        (SDRLSR, ["size^3.1"]),
        (SDRLSM, ["size^3.1"]),
        (SAOneLambda, ["size", 2, 2]),
        (MuPlusOneInverseK, ["size^1.2"]),
        (cGA, ["sqrt*ln"])
    ]

    algorithm = algorithms[index_algorithm][0]
    parameters = algorithms[index_algorithm][1]

    print(algorithm.name + ";" + str(parameters) + ";")
    header = "Sat problem;Number of variables;Number of clauses;Parameters;"
    for i in range(1, runs+1):
        header += str(i) + ";"
    print(header)

    sat_problems = sats(folder_name)
    for sat_problem in sat_problems:
        for i in range(len(parameters)):
            algorithm.parameters[i].default_value = parameters[i]

        parameters_instance = default_parameters(algorithm, sat_problem.number_of_variables)
        line = sat_problem.name + ";" \
               + str(sat_problem.number_of_variables) + ";" \
               + str(sat_problem.number_of_clauses) + ";" \
               + str(parameters_instance) + ";"

        results = parallelize_sat(algorithm, parameters_instance, sat_problem, runs, max_iter, nb_cores)

        for i in range(len(results)):
            line += str(results[i]) + ";"
        print(line)


def parallelize_sat(evolutionary_algorithm, evolutionary_parameters, sat_problem, runs, max_iter, nb_cores):
    pool = multiprocessing.Pool(nb_cores)
    results = pool.map(functools.partial(partial_sat,
                       evolutionary_algorithm, evolutionary_parameters, sat_problem, max_iter), range(runs))
    pool.close()
    return np.array(results)


# Function solving an evolutionary algorithm on a fitness function and
# returning the number of call to the fitness function
def partial_sat(evolutionary_algorithm, evolutionary_parameter_values, sat_problem, max_iter, i):
    _, iterations, _, _, _ = evolutionary_algorithm.solve_SAT(evolutionary_parameter_values,
                                                              sat_problem,
                                                              False, max_iter)
    return iterations

