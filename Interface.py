import tkinter

from Action import *
from InterfaceUtilities import *


# Total number of cores on the computer
cores = int(multiprocessing.cpu_count()/2)


# Parent class for all the interfaces
class Interface(tk.Frame):

    def __init__(self, class_name, parent, controller):
        self.class_name = class_name
        self.parent = parent
        self.controller = controller


# Parent class for the different action interfaces
class ActionInterface(Interface):

    def __init__(self, class_name, parent, controller, action,
                 problem_size, problem_size_end, step, iterations,
                 fitness_function_name, fitness_parameter_values,
                 evolutionary_algorithm_name, evolutionary_parameter_values,
                 sat_file_name, sat_folder_name,
                 labels, xs, ys, yss):

        super().__init__(class_name, parent, controller)
        self.action = action
        self.fitness_function_name = fitness_function_name
        self.fitness_function = find_fitness(fitness_function_name.get())
        self.fitness_parameters = self.fitness_function.parameters
        self.fitness_parameter_values = fitness_parameter_values
        self.evolutionary_algorithm_name = evolutionary_algorithm_name
        self.evolutionary_algorithm = find_evolutionary(evolutionary_algorithm_name.get())
        self.evolutionary_parameters = self.evolutionary_algorithm.parameters
        self.evolutionary_parameter_values = evolutionary_parameter_values
        self.sat_file_name = sat_file_name
        self.sat_problem = SAT(sat_file_name)
        self.sat_folder_name = sat_folder_name
        self.sat_problems = sats(sat_folder_name)
        self.problem_size = problem_size
        self.problem_size_end = problem_size_end
        self.step = step
        self.iterations = iterations
        self.labels = labels
        self.xs = xs
        self.ys = ys
        self.yss = yss

        frame_creation(self, action.description, StartPage)

    # Entry for length of the bit strings
    def choice_of_problem_size(self, start_row):
        tk.Label(self, text="Problem Size:").grid(row=start_row, column=0, padx=5, pady=5)
        tk.Entry(self, justify=CENTER, textvariable=self.problem_size).grid(row=start_row, column=1, padx=5, pady=5)
        # Returning the next free row
        return start_row + 1

    # Choice of a range of length for the bit strings for statistical analyses
    def choice_of_problem_range(self, start_row):
        tk.Label(self, text="Problem size from _ to _:").grid(row=start_row, column=0, padx=5, pady=5)
        tk.Entry(self, justify=CENTER, textvariable=self.problem_size).grid(row=start_row, column=1, padx=5, pady=5)
        tk.Entry(self, justify=CENTER, textvariable=self.problem_size_end).grid(row=start_row, column=2, padx=5, pady=5)
        tk.Label(self, text="Step:").grid(row=start_row+1, column=0, padx=5, pady=5)
        tk.Entry(self, justify=CENTER, textvariable=self.step).grid(row=start_row+1, column=1, padx=5, pady=5)
        # Returning the next free row
        return start_row + 2

    # Choice of number of run of an algorithm on a fitness function for statistical analyses
    def choice_of_number_of_iterations(self, start_row):
        tk.Label(self, text="Number of iterations:").grid(row=start_row, column=0, padx=5, pady=5)
        tk.Entry(self, justify=CENTER, textvariable=self.iterations).grid(row=start_row, column=1, padx=5, pady=5)
        # Returning the next free row
        return start_row + 1

    # Creation of a new frame with the new fitness keeping the previous parameters
    def change_fitness(self, *args):
        # Find the new fitness function from dropdown menu
        fitness_function = find_fitness(self.fitness_function_name.get())
        new_fitness_name = StringVar()
        new_fitness_name.set(self.fitness_function_name.get())
        # Creation of the new frame with the new fitness function and its parameters set to default
        new_frame = self.class_name(self.class_name, self.parent, self.controller, self.action,
                                    self.problem_size, self.problem_size_end, self.step, self.iterations,
                                    new_fitness_name,
                                    default_parameters(fitness_function, self.problem_size.get()),
                                    self.evolutionary_algorithm_name,
                                    self.evolutionary_parameter_values,
                                    self.sat_file_name, self.sat_folder_name,
                                    [], [], [], [])
        new_frame.grid(row=0, column=0, sticky="nsew")
        # Display this new frame
        new_frame.tkraise()
        # Reset name for consistency
        self.fitness_function_name.set(fitness_function_names[0])

    # Dropdown menu for choosing a fitness function
    def choice_of_fitness_function(self, start_row):
        tk.Label(self, text="Choose a fitness function: ").grid(row=start_row, column=0, padx=5, pady=5)
        choose_fitness = OptionMenu(self,
                                    self.fitness_function_name,
                                    *fitness_function_names,
                                    command=self.change_fitness)
        choose_fitness.grid(row=start_row, column=1, padx=5, pady=5)
        # Returning the next free row
        return start_row + 1

    # Function creating the entry for the parameters of the fitness function
    def choice_of_fitness_parameters(self, start_row):
        if len(self.fitness_parameters) == 0:
            # Returning the next free row
            return start_row
        else:
            tk.Label(self, text="Choose parameters for the fitness function:")\
                .grid(row=start_row, column=0, padx=5, pady=5)
            for i in range(len(self.fitness_parameters)):
                tk.Label(self, text=self.fitness_parameters[i].name).grid(row=start_row+i, column=1, padx=5, pady=5)
                tk.Entry(self, justify=CENTER, textvariable=self.fitness_parameter_values[i])\
                    .grid(row=start_row+i, column=2, padx=5, pady=5)
            # Returning the next free row
            return start_row + len(self.fitness_parameters)

    # Creation of a new frame with the new algorithm keeping the previous parameters
    def change_evolutionary(self, *args):
        # Find the new evolutionary algorithm from dropdown menu
        evolutionary_algorithm = find_evolutionary(self.evolutionary_algorithm_name.get())
        new_evolutionary_name = StringVar()
        new_evolutionary_name.set(self.evolutionary_algorithm_name.get())
        # Creation of the new frame with the new evolutionary algorithm and its parameters set to default
        new_frame = self.class_name(self.class_name, self.parent, self.controller, self.action,
                                    self.problem_size, self.problem_size_end, self.step, self.iterations,
                                    self.fitness_function_name, self.fitness_parameter_values,
                                    new_evolutionary_name,
                                    default_parameters(evolutionary_algorithm, self.problem_size.get()),
                                    self.sat_file_name, self.sat_folder_name,
                                    self.labels, self.xs, self.ys, self.yss)
        new_frame.grid(row=0, column=0, sticky="nsew")
        new_frame.tkraise()
        # Reset name for consistency
        self.evolutionary_algorithm_name.set(evolutionary_algorithm_names[0])

    # Dropdown menu for choosing an evolutionary algorithm
    def choice_of_evolutionary_algorithm(self, start_row):
        tk.Label(self, text="Choose an evolutionary algorithm: ").grid(row=start_row, column=0, padx=5, pady=5)
        choose_algorithm = OptionMenu(self,
                                      self.evolutionary_algorithm_name,
                                      *evolutionary_algorithm_names,
                                      command=self.change_evolutionary)
        choose_algorithm.grid(row=start_row, column=1, padx=5, pady=5)
        # Returning the next free row
        return start_row + 1

    # Function creating the entry for the parameters of the evolutionary algorithm
    def choice_of_evolutionary_parameters(self, start_row):
        if len(self.evolutionary_parameters) == 0:
            # Returning the next free row
            return start_row
        else:
            tk.Label(self, text="Choose parameters for the evolutionary algorithm:")\
                .grid(row=start_row, column=0, padx=5, pady=5)
            for i in range(len(self.evolutionary_parameters)):
                tk.Label(self, text=self.evolutionary_parameters[i].name)\
                    .grid(row=start_row+i, column=1, padx=5, pady=5)
                tk.Entry(self, justify=CENTER, textvariable=self.evolutionary_parameter_values[i])\
                    .grid(row=start_row+i, column=2, padx=5, pady=5)
            # Returning the next free row
            return start_row + len(self.evolutionary_parameters)

    # Function creating the possibility to browse a file for chOosing a SAT problem
    def choice_of_sat_file(self, start_row):
        tkinter.Label(self, text="Choose a MAX-SAT problem:")\
            .grid(row=start_row, column=0, padx=5, pady=5)
        tkinter.Label(self, text=self.sat_problem.name)\
            .grid(row=start_row, column=1, padx=5, pady=5)

        # Browse file button
        button = ttk.Button(self, text="Browse file", command=lambda: self.get_file_name(start_row))
        button.grid(row=start_row, column=2, padx=5, pady=5)

        return start_row + 1

    # Function getting a .cnf file that was selected and creating a SAT instance
    def get_file_name(self, row):
        file_name = filedialog.askopenfilename(initialdir="cnf",
                                               title="Select a File",
                                               filetypes=([("CNF files", "*.cnf")])
                                               )
        self.sat_file_name = file_name
        self.sat_problem = SAT(self.sat_file_name)

        for label in self.grid_slaves():
            if int(label.grid_info()["row"]) == row and  int(label.grid_info()["column"]) == 1:
                label.grid_forget()
        tkinter.Label(self, text=self.sat_problem.name) \
            .grid(row=row, column=1, padx=5, pady=5)

    # Function creating the possibility to browse a file a folder for selection several SAT instances
    def choice_of_sat_folder(self, start_row):
        tkinter.Label(self, text="Choose folder of MAX-SAT problems:") \
            .grid(row=start_row, column=0, padx=5, pady=5)
        tkinter.Label(self, text=self.sat_folder_name) \
            .grid(row=start_row, column=1, padx=5, pady=5)

        # Browse file button
        button = ttk.Button(self, text="Browse folder", command=lambda: self.get_folder_name(start_row))
        button.grid(row=start_row, column=2, padx=5, pady=5)

        return start_row + 1

    # Function getting a folder and creating a SAT instance for each of the .cnf files inside
    def get_folder_name(self, row):
        folder_name = filedialog.askdirectory(initialdir="cnf",
                                              title="Select a folder"
                                              )
        self.sat_folder_name = os.path.basename(folder_name)
        self.sat_problems = sats(folder_name)

        for label in self.grid_slaves():
            if int(label.grid_info()["row"]) == row and int(label.grid_info()["column"]) == 1:
                label.grid_forget()
        tkinter.Label(self, text=self.sat_folder_name) \
            .grid(row=row, column=1, padx=5, pady=5)

    # Checking that the length is consistent with the constraint of the fitness function
    # Display constraint if not respected / Erase if respected
    def check_size(self, size_row, size_column):
        valid = True
        size_constraint = self.fitness_function.size_constraint
        # erasing previous information message
        for label in self.grid_slaves(size_row, size_column):
            label.grid_forget()
        # If a constraint is not respected, display it to the user
        if not size_constraint.check_condition(self.problem_size.get()):
            tk.Label(self, text=size_constraint.description).grid(row=size_row, column=size_column, padx=5, pady=5)
            valid = False
        # Return true if all the constraint are respected
        return valid

    # Checking that the length range is consistent with the constraint of the fitness function
    # Display constraint if not respected / Erase if respected
    def check_range(self, size_row, size_column, step_row, step_column):
        valid = True
        size_constraint = self.fitness_function.size_constraint
        # erasing previous information message
        for label in self.grid_slaves(size_row, size_column) + self.grid_slaves(step_row, step_column):
            label.grid_forget()
        # Check that the range is correct
        if not self.problem_size.get() <= self.problem_size_end.get():
            tk.Label(self, text="Incorrect range").grid(row=size_row, column=size_column, padx=5, pady=5)
            valid = False
        # If a constraint is not respected, display it to the user
        elif not size_constraint.check_condition(self.problem_size.get()):
            tk.Label(self, text=size_constraint.description).grid(row=size_row, column=size_column, padx=5, pady=5)
            valid = False
        if not size_constraint.check_condition(self.step.get()):
            tk.Label(self, text=size_constraint.description).grid(row=step_row, column=step_column, padx=5, pady=5)
            valid = False
        # Return true if all the constraint are respected
        return valid

    # Check that the fitness parameters respect all the constraint
    # Display constraint if not respected / Erase if respected
    def check_fitness_parameters(self, fitness_row, fitness_column, check_end):
        valid = True
        fitness_parameters = self.fitness_function.parameters
        for i in range(len(fitness_parameters)):
            # erasing previous information message
            for label in self.grid_slaves(fitness_row + i, fitness_column):
                label.grid_forget()
            correctness = fitness_parameters[i].is_value_valid(self.fitness_parameter_values[i].get(),
                                                               self.problem_size.get(), self.problem_size_end.get(),
                                                               check_end)
            # If a constraint is not respected, display it to the user
            if correctness != "correct":
                tk.Label(self, text=correctness).grid(row=fitness_row + i, column=fitness_column, padx=5, pady=5)
                valid = False
        # Return true if all the constraint are respected
        return valid

    # Check that the algorithm parameters respect all the constraint
    # Display constraint if not respected / Erase if respected
    def check_evolutionary_parameters(self, evolutionary_row, evolutionary_column, check_end):
        valid = True
        evolutionary_parameters = self.evolutionary_algorithm.parameters
        for i in range(len(evolutionary_parameters)):
            # erasing previous information message
            for label in self.grid_slaves(evolutionary_row + i, evolutionary_column):
                label.grid_forget()
            correctness = evolutionary_parameters[i].is_value_valid(self.evolutionary_parameter_values[i].get(),
                                                                    self.problem_size.get(),
                                                                    self.problem_size_end.get(),
                                                                    check_end)
            # If a constraint is not respected, display it to the user
            if correctness != "correct":
                tk.Label(self, text=correctness)\
                    .grid(row=evolutionary_row + i, column=evolutionary_column, padx=5, pady=5)
                valid = False
        # Return true if all the constraint are respected
        return valid


# Class for the main page interface
class StartPage(Interface):
    def __init__(self, class_name, parent, controller):
        super().__init__(class_name, parent, controller)

        # Creating of the frame
        frame_creation(self, "Main page")

        # Creating a button for each action leading to their interface
        tk.Label(self, text="Choose an action: ").grid(row=2, column=0, padx=5, pady=5)

        for i in range(len(actions)):
            ttk.Button(self, text=actions[i].description,
                       command=lambda i=i: controller.show_frame(globals()[actions[i].name]))\
                .grid(row=i+3, column=1, padx=5, pady=5)


# Interface for generating a graph of a fitness function
class DF(ActionInterface):

    def __init__(self, class_name, parent, controller, action,
                 problem_size, problem_size_end, step, iterations,
                 fitness_function, fitness_parameter_values,
                 evolutionary_algorithm, evolutionary_parameter_values,
                 sat_file_name, sat_folder_name,
                 labels, xs, ys, yss):

        super().__init__(class_name, parent, controller, action,
                         problem_size, problem_size_end, step, iterations,
                         fitness_function, fitness_parameter_values,
                         evolutionary_algorithm, evolutionary_parameter_values,
                         sat_file_name, sat_folder_name,
                         labels, xs, ys, yss)

        row = self.choice_of_problem_size(2)
        fitness_row = self.choice_of_fitness_function(row)
        row = self.choice_of_fitness_parameters(fitness_row)

        # Create the graph of the fitness function
        display_button = ttk.Button(self, text="Display graph",
                                    command=lambda: self.display_fitness_plot(row+1, 2, 2, fitness_row, 3))
        display_button.grid(row=row, column=2, padx=5, pady=5)

    # Display the fitness function as function of the norm of the bitstring
    def display_fitness_plot(self, start_row, size_row, size_column, fitness_row, fitness_column):
        problem_size = self.problem_size.get()
        # Check that all the parameters are ok
        if self.check_size(size_row, size_column) and self.check_fitness_parameters(fitness_row, fitness_column, False):
            fitness_parameter_values = []
            for fitness_parameter_value in self.fitness_parameter_values:
                fitness_parameter_values.append(fitness_parameter_value.get())
            bit_string = only_zeros(problem_size)
            x = np.empty(problem_size + 1)
            y = np.empty(problem_size + 1)

            x[0] = 0
            y[0] = self.fitness_function.result(fitness_parameter_values, bit_string)
            # Create the plot by adding one zero to the bit string at each iteration (ie +1 to the norm)
            for i in range(problem_size):
                bit_string.add_one_one()
                x[i + 1] = i + 1
                y[i + 1] = self.fitness_function.result(fitness_parameter_values, bit_string)
            build_plot(self, [("|x|", "f(|x|)", self.fitness_function_name.get())], [x], [y], start_row, 2,
                       self.fitness_function.name + " as a function of the norm", '|x|', 'f(x)')


# Interface displaying a run of an evolutionary algorithm on a fitness function
class R1(ActionInterface):

    def __init__(self, class_name, parent, controller, action,
                 problem_size, problem_size_end, step, iterations,
                 fitness_function, fitness_parameter_values,
                 evolutionary_algorithm, evolutionary_parameter_values,
                 sat_file_name, sat_folder_name,
                 labels, xs, ys, yss):

        super().__init__(class_name, parent, controller, action,
                         problem_size, problem_size_end, step, iterations,
                         fitness_function, fitness_parameter_values,
                         evolutionary_algorithm, evolutionary_parameter_values,
                         sat_file_name, sat_folder_name,
                         labels, xs, ys, yss)

        row = self.choice_of_problem_size(2)
        fitness_row = self.choice_of_fitness_function(row)
        row = self.choice_of_fitness_parameters(fitness_row)
        evolutionary_row = self.choice_of_evolutionary_algorithm(row)
        row = self.choice_of_evolutionary_parameters(evolutionary_row)

        # Solve the problem and display results
        display_button = ttk.Button(self, text="Run", command=lambda: self.solve(row+1, 2, 2,
                                                                                 fitness_row, 3, evolutionary_row, 3))
        display_button.grid(row=row, column=2, padx=5, pady=5)

    # Solving the fitness function using one evolutionary algorithm
    def solve(self, start_row, size_row, size_column,
              fitness_row, fitness_column,
              evolutionary_row, evolutionary_column):
        # Check that all the parameters are ok
        if self.check_size(size_row, size_column) and \
                self.check_fitness_parameters(fitness_row, fitness_column, False) and \
                self.check_evolutionary_parameters(evolutionary_row, evolutionary_column, False):
            problem_size = self.problem_size.get()
            fitness_parameter_values = []
            for fitness_parameter_value in self.fitness_parameter_values:
                fitness_parameter_values.append(fitness_parameter_value.get())
            evolutionary_parameter_values = []
            for evolutionary_parameter_value in self.evolutionary_parameter_values:
                evolutionary_parameter_values.append(evolutionary_parameter_value.get())

            # Running the evolutionary algorithm on the fitness function
            bit_string, iterations, timer, x, y = self.evolutionary_algorithm.solve_fitness(
                evolutionary_parameter_values, problem_size, self.fitness_function, fitness_parameter_values, True)
            tk.Label(self, text="The solution was found in " + str(round(timer, 2)) + " seconds")\
                .grid(row=start_row, column=1, padx=5, pady=5)
            tk.Label(self, text="The solution was found in " + str(iterations) + " iterations")\
                .grid(row=start_row+1, column=1, padx=5, pady=5)

            build_plot(self, [("Iteration", self.fitness_function_name.get(), self.evolutionary_algorithm.name)],
                       [x], [y], start_row+2, 1, "Improvements of the bit string", 'iterations', 'f(x)')


# Interface for running an evolutionary algorithm n times on a fitness function and returning statistics
class RN(ActionInterface):

    def __init__(self, class_name, parent, controller, action,
                 problem_size, problem_size_end, step, iterations,
                 fitness_function, fitness_parameter_values,
                 evolutionary_algorithm, evolutionary_parameter_values,
                 sat_file_name, sat_folder_name,
                 labels, xs, ys, yss):

        super().__init__(class_name, parent, controller, action,
                         problem_size, problem_size_end, step, iterations,
                         fitness_function, fitness_parameter_values,
                         evolutionary_algorithm, evolutionary_parameter_values,
                         sat_file_name, sat_folder_name,
                         labels, xs, ys, yss)

        row = self.choice_of_problem_size(2)
        row = self.choice_of_number_of_iterations(row)
        fitness_row = self.choice_of_fitness_function(row)
        row = self.choice_of_fitness_parameters(fitness_row)
        evolutionary_row = self.choice_of_evolutionary_algorithm(row)
        row = self.choice_of_evolutionary_parameters(evolutionary_row)

        # Create the graph of the fitness function
        display_button = ttk.Button(self, text="Run", command=lambda: self.solve_n_times(row+1, 2, 2,
                                                                                         fitness_row, 3,
                                                                                         evolutionary_row, 3))
        display_button.grid(row=row, column=2, padx=5, pady=5)

    # Running the evolutionary algorithm n times on a fitness function and returning statistical analysis
    def solve_n_times(self, start_row, size_row, size_column,
                      fitness_row, fitness_column, evolutionary_row, evolutionary_column):
        # Check that all the parameters are ok
        if self.check_size(size_row, size_column) and \
                self.check_fitness_parameters(fitness_row, fitness_column, False) and \
                self.check_evolutionary_parameters(evolutionary_row, evolutionary_column, False):
            problem_size = self.problem_size.get()
            iterations = self.iterations.get()
            fitness_parameter_values = []
            for fitness_parameter_value in self.fitness_parameter_values:
                fitness_parameter_values.append(fitness_parameter_value.get())
            evolutionary_parameter_values = []
            for evolutionary_parameter_value in self.evolutionary_parameter_values:
                evolutionary_parameter_values.append(evolutionary_parameter_value.get())
            # Adding the number of iterations of each run in an array
            results = run_parallel(iterations, problem_size, self.evolutionary_algorithm, evolutionary_parameter_values,
                                   self.fitness_function, fitness_parameter_values, cores)
            tk.Label(self, text="The minimum number of iterations is: " + str(int(results.min()))) \
                .grid(row=start_row, column=1, padx=5, pady=5)
            tk.Label(self, text="The maximum number of iterations is: " + str(int(results.max()))) \
                .grid(row=start_row + 1, column=1, padx=5, pady=5)
            tk.Label(self, text="The mean of the number of iterations is: " + str(round(results.mean(), 2))) \
                .grid(row=start_row + 2, column=1, padx=5, pady=5)
            tk.Label(self, text="The standard deviation of the number of iterations is: " +
                                str(round(np.std(results), 2))) \
                .grid(row=start_row + 3, column=1, padx=5, pady=5)
            tk.Label(self, text="The median of the number of iterations is: " + str(int(np.median(results)))) \
                .grid(row=start_row + 4, column=1, padx=5, pady=5)

            extract_full_data_button(self, [self.fitness_function.name], [[problem_size]], [[results]],
                                     start_row + 5, 1)


# Interface for running an evolutionary algorithm n times on a fitness function for a range of length
class RNM(ActionInterface):

    def __init__(self, class_name, parent, controller, action,
                 problem_size, problem_size_end, step, iterations,
                 fitness_function, fitness_parameter_values,
                 evolutionary_algorithm, evolutionary_parameter_values,
                 sat_file_name, sat_folder_name,
                 labels, xs, ys, yss):

        super().__init__(class_name, parent, controller, action,
                         problem_size, problem_size_end, step, iterations,
                         fitness_function, fitness_parameter_values,
                         evolutionary_algorithm, evolutionary_parameter_values,
                         sat_file_name, sat_folder_name,
                         labels, xs, ys, yss)

        row = self.choice_of_problem_range(2)
        row = self.choice_of_number_of_iterations(row)
        fitness_row = self.choice_of_fitness_function(row)
        row = self.choice_of_fitness_parameters(fitness_row)
        evolutionary_row = self.choice_of_evolutionary_algorithm(row)
        row = self.choice_of_evolutionary_parameters(evolutionary_row)

        # Create the graphs of the fitness function
        display_button = ttk.Button(self, text="Run", command=lambda: self.solve_n_m_times(row+1, 2, 3, 3, 2,
                                                                                           fitness_row, 3,
                                                                                           evolutionary_row, 3))
        display_button.grid(row=row, column=2, padx=5, pady=5)

    # Function solving the problem for a range different problem size and returning a graph of the means
    def solve_n_m_times(self, start_row, size_row, size_column, step_row, step_column, fitness_row, fitness_column,
                        evolutionary_row, evolutionary_column):
        if self.check_range(size_row, size_column, step_row, step_column) and \
                self.check_fitness_parameters(fitness_row, fitness_column, True) and \
                self.check_evolutionary_parameters(evolutionary_row, evolutionary_column, True):
            problem_size = self.problem_size.get()
            problem_size_end = self.problem_size_end.get()
            step = self.step.get()
            iterations = self.iterations.get()
            fitness_parameter_values = []
            for fitness_parameter_value in self.fitness_parameter_values:
                fitness_parameter_values.append(fitness_parameter_value.get())
            evolutionary_parameter_values = []
            for evolutionary_parameter_value in self.evolutionary_parameter_values:
                evolutionary_parameter_values.append(evolutionary_parameter_value.get())
            x = []
            y = []
            y_box_plot = []
            # For every length in the range, adding the mean to the plot and the iteration values to create a boxplot
            for i in range(problem_size, problem_size_end+1, step):
                print("Problem size: " + str(i))
                results = run_parallel(iterations, i, self.evolutionary_algorithm, evolutionary_parameter_values,
                                       self.fitness_function, fitness_parameter_values, cores)
                x.append(i)
                y.append(round(results.mean(), 0))
                y_box_plot.append(results)
            print("Problem size: Done")

            build_plot(self, [("Problem size", self.fitness_function_name.get(), self.evolutionary_algorithm_name.get())],
                       [x], [y], start_row, 0, "Plot of the mean of the runs as a function of the problem size",
                       "Problem size", "Mean of the runs")
            build_box_plot(self, self.evolutionary_algorithm.name, x, y_box_plot, start_row, 2,
                           "Box plot of the number of iterations as a function of the problem size",
                           "Problem_size", "Iterations")


# Interface for running evolutionary algorithms n times on a fitness function for a range of length and comparing them
class RKNM(ActionInterface):

    def __init__(self, class_name, parent, controller, action,
                 problem_size, problem_size_end, step, iterations,
                 fitness_function, fitness_parameter_values,
                 evolutionary_algorithm, evolutionary_parameter_values,
                 sat_file_name, sat_folder_name,
                 labels, xs, ys, yss):

        super().__init__(class_name, parent, controller, action,
                         problem_size, problem_size_end, step, iterations,
                         fitness_function, fitness_parameter_values,
                         evolutionary_algorithm, evolutionary_parameter_values,
                         sat_file_name, sat_folder_name,
                         labels, xs, ys, yss)

        row = self.choice_of_problem_range(2)
        row = self.choice_of_number_of_iterations(row)
        fitness_row = self.choice_of_fitness_function(row)
        row = self.choice_of_fitness_parameters(fitness_row)
        evolutionary_row = self.choice_of_evolutionary_algorithm(row)
        row = self.choice_of_evolutionary_parameters(evolutionary_row)

        # Create the graph of the fitness function
        display_button = ttk.Button(self, text="New run", command=lambda: self.new_run(row+1, 2, 3, 3, 2,
                                                                                       fitness_row, 3,
                                                                                       evolutionary_row, 3))
        display_button.grid(row=row, column=2, padx=5, pady=5)

        # If there is already one graph displayed, add a button in order to add a new algorithm to same graph
        if len(self.labels) > 0:
            # Add a new run from the previous graph
            display_button = ttk.Button(self, text="Add run",
                                        command=lambda: self.solve_k_n_m_times(row+1, 2, 3, 3, 2,
                                                                               fitness_row, 3,
                                                                               evolutionary_row, 3))
            display_button.grid(row=row, column=3, padx=5, pady=5)

    # Function running the EA for a range of problem length and getting statistical results
    # Add the plot to the previous plots if there was already
    def solve_k_n_m_times(self, start_row, size_row, size_column, step_row, step_column, fitness_row, fitness_column,
                          evolutionary_row, evolutionary_column):
        if self.check_range(size_row, size_column, step_row, step_column) and \
                self.check_fitness_parameters(fitness_row, fitness_column, True) and \
                self.check_evolutionary_parameters(evolutionary_row, evolutionary_column, True):
            problem_size = self.problem_size.get()
            problem_size_end = self.problem_size_end.get()
            step = self.step.get()

            iterations = self.iterations.get()
            fitness_parameter_values = []
            for fitness_parameter_value in self.fitness_parameter_values:
                fitness_parameter_values.append(fitness_parameter_value.get())
            evolutionary_parameter_values = []
            for evolutionary_parameter_value in self.evolutionary_parameter_values:
                evolutionary_parameter_values.append(evolutionary_parameter_value.get())
            x = []
            y = []
            ys = []
            # For every length in the range it adds the mean to the plot
            for i in range(problem_size, problem_size_end + 1, step):
                print("Problem size: " + str(i))
                results = run_parallel(iterations, i, self.evolutionary_algorithm, evolutionary_parameter_values,
                                       self.fitness_function, fitness_parameter_values, cores)
                x.append(i)
                y.append(round(results.mean(), 0))
                ys.append(results)
            print("Problem size: Done")
            # Add the plot to the already computed ones
            self.labels.append(("Iterations", self.fitness_function_name.get(), self.evolutionary_algorithm_name.get()))
            self.xs.append(x)
            self.ys.append(y)
            self.yss.append(ys)

            sheet_names = []
            for i in range(len(self.labels)):
                sheet_names.append(self.labels[i][2])

            extract_full_data_button(self, sheet_names, self.xs, self.yss, start_row-1, 1)

            build_plot(self, self.labels, self.xs, self.ys, start_row, 2,
                       "Comparison of different algorithms on " + self.fitness_function_name.get(),
                       "Problem size", "Mean of the runs")

            # Add a button for adding a new run from the previous graph
            display_button = ttk.Button(self, text="Add run",
                                        command=lambda: self.solve_k_n_m_times(start_row, size_row, size_column, step_row, step_column, fitness_row, fitness_column,
                                                                               evolutionary_row, evolutionary_column))
            display_button.grid(row=start_row-1, column=3, padx=5, pady=5)

    # Erase the previous plot to start a new one
    def new_run(self, start_row, size_row, size_column, step_row, step_column, fitness_row, fitness_column,
                evolutionary_row, evolutionary_column):
        self.labels = []
        self.xs = []
        self.ys = []
        self.yss = []
        self.solve_k_n_m_times(start_row, size_row, size_column, step_row, step_column, fitness_row, fitness_column,
                               evolutionary_row, evolutionary_column)


# Interface for running an algorithm on a SAT instance and displaying the run
class ISAT(ActionInterface):

    def __init__(self, class_name, parent, controller, action,
                 problem_size, problem_size_end, step, iterations,
                 fitness_function, fitness_parameter_values,
                 evolutionary_algorithm, evolutionary_parameter_values,
                 sat_file_name, sat_folder_name,
                 labels, xs, ys, yss):

        super().__init__(class_name, parent, controller, action,
                         problem_size, problem_size_end, step, iterations,
                         fitness_function, fitness_parameter_values,
                         evolutionary_algorithm, evolutionary_parameter_values,
                         sat_file_name, sat_folder_name,
                         labels, xs, ys, yss)

        evolutionary_row = self.choice_of_evolutionary_algorithm(2)
        row = self.choice_of_evolutionary_parameters(evolutionary_row)
        row = self.choice_of_sat_file(row)

        # Solve the problem and display results
        display_button = ttk.Button(self, text="Run", command=lambda: self.solve(row + 1, evolutionary_row, 3))
        display_button.grid(row=row, column=2, padx=5, pady=5)

    def solve(self, start_row,
              evolutionary_row, evolutionary_column):
        # Check that all the parameters are ok
        if self.check_evolutionary_parameters(evolutionary_row, evolutionary_column, False):
            evolutionary_parameter_values = []
            for evolutionary_parameter_value in self.evolutionary_parameter_values:
                evolutionary_parameter_values.append(evolutionary_parameter_value.get())

            # Running the evolutionary algorithm on the fitness function
            bit_string, iterations, timer, x, y = self.evolutionary_algorithm.solve_SAT(
                evolutionary_parameter_values, self.sat_problem, True)
            tk.Label(self, text="The solution was found in " + str(round(timer, 2)) + " seconds") \
                .grid(row=start_row, column=1, padx=5, pady=5)
            tk.Label(self, text="The solution was found in " + str(iterations) + " iterations") \
                .grid(row=start_row + 1, column=1, padx=5, pady=5)
            tk.Label(self, text="The solution found is " + bit_string.string) \
                .grid(row=start_row + 2, column=1, padx=5, pady=5)

            build_plot(self, [("Iteration", self.sat_problem.name, self.evolutionary_algorithm.name)],
                       [x], [y], start_row + 3, 1, "Improvements of the bit string", 'iterations', 'f(x)')


# Interface for running an algorithm on several SAT instances and returning statistical results
class ISATS(ActionInterface):
    def __init__(self, class_name, parent, controller, action,
                 problem_size, problem_size_end, step, iterations,
                 fitness_function, fitness_parameter_values,
                 evolutionary_algorithm, evolutionary_parameter_values,
                 sat_file_name, sat_folder_name,
                 labels, xs, ys, yss):

        super().__init__(class_name, parent, controller, action,
                         problem_size, problem_size_end, step, iterations,
                         fitness_function, fitness_parameter_values,
                         evolutionary_algorithm, evolutionary_parameter_values,
                         sat_file_name, sat_folder_name,
                         labels, xs, ys, yss)

        evolutionary_row = self.choice_of_evolutionary_algorithm(2)
        row = self.choice_of_evolutionary_parameters(evolutionary_row)
        row = self.choice_of_sat_folder(row)

        # Solve the problem and display results
        display_button = ttk.Button(self, text="Run", command=lambda: self.solve(row + 1, evolutionary_row, 3))
        display_button.grid(row=row, column=2, padx=5, pady=5)

    # Solving the SAT instances using one evolutionary algorithm
    def solve(self, start_row,
              evolutionary_row, evolutionary_column):
        # Check that all the parameters are ok
        if self.check_evolutionary_parameters(evolutionary_row, evolutionary_column, False):
            evolutionary_parameter_values = []
            for evolutionary_parameter_value in self.evolutionary_parameter_values:
                evolutionary_parameter_values.append(evolutionary_parameter_value.get())

            results = run_parallel_sat(self.evolutionary_algorithm, evolutionary_parameter_values,
                                       self.sat_problems, cores)

            self.xs = []
            self.ys = np.empty(len(results))

            for i in range(len(results)):
                self.xs.append(results[i][0])
                self.ys[i] = results[i][1]

            # Find a better way of representing things here
            tk.Label(self, text="The minimum number of iterations is: " + str(int(self.ys.min()))) \
                .grid(row=start_row, column=1, padx=5, pady=5)
            tk.Label(self, text="The maximum number of iterations is: " + str(int(self.ys.max()))) \
                .grid(row=start_row + 1, column=1, padx=5, pady=5)
            tk.Label(self, text="The mean of the number of iterations is: " + str(round(self.ys.mean(), 2))) \
                .grid(row=start_row + 2, column=1, padx=5, pady=5)
            tk.Label(self, text="The standard deviation of the number of iterations is: " +
                                str(round(np.std(self.ys), 2))) \
                .grid(row=start_row + 3, column=1, padx=5, pady=5)
            tk.Label(self, text="The median of the number of iterations is: " + str(int(np.median(self.ys)))) \
                .grid(row=start_row + 4, column=1, padx=5, pady=5)

            build_plot(self, [("File Name", "iterations", self.evolutionary_algorithm.name)],
                       [self.xs], [self.ys], start_row + 5, 1, "Results", 'File name', 'iterations')
