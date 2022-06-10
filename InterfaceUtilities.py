import multiprocessing
import sys

import pandas as pd
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from datetime import datetime

from EvolutionaryAlgorithm import *
from FitnessFunction import *
from SAT import *


# Create frame with return button if not the main page
def frame_creation(root, title, start_page=False):
    # Creation of the frame
    tk.Frame.__init__(root, root.parent)

    # Title of the page
    label = tk.Label(root, text=title, font=("Arial", 20))
    label.grid(row=0, column=0, columnspan=4, padx=5, pady=5)

    # Add return button
    if start_page:
        # Button back to main page
        button = ttk.Button(root, text="Return to main page",
                            command=lambda: root.controller.show_frame(start_page))
        button.grid(row=0, column=4, padx=5, pady=5)


# Return the evolutionary algorithm with the input name
def find_evolutionary(evolutionary_name):
    for evolutionary_algorithm in evolutionary_algorithms:
        if evolutionary_algorithm.name == evolutionary_name:
            return evolutionary_algorithm
    raise Exception("Evolutionary algorithm not found")


# Return the fitness function with the input name
def find_fitness(fitness_name):
    for fitness_function in fitness_functions:
        if fitness_function.name == fitness_name:
            return fitness_function
    raise Exception("Fitness function not found")


# Build a graph with the x and y values
def build_plot(root, labels, xs, ys, row, column, title, x_label, y_label):
    fig = Figure(figsize=(6, 4), dpi=100)
    a = fig.add_subplot(111)

    # Setting the title and axis labels
    a.set_title(title)
    a.set_xlabel(x_label)
    a.set_ylabel(y_label)

    # Plotting all the plots, adding the label names
    for i in range(len(labels)):
        a.plot(xs[i], ys[i], '.', label=labels[i][2])

    # If there is more than one plot, the legend is displayed on the left corner
    if len(labels) > 1:
        a.legend(loc='upper left', frameon=False)

    # Creation of the canvas
    canvas = FigureCanvasTkAgg(fig, root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=row, column=column, columnspan=2, padx=5, pady=5)

    # Creation of the toolbar, displayed under the graph
    toolbar_frame = Frame(root)
    toolbar_frame.grid(row=row+1, column=column)
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()
    canvas.get_tk_widget().grid(row=row, column=column, columnspan=2, padx=5, pady=5)

    # Button to extract data
    button = ttk.Button(root, text="Export data",
                        command=lambda: extract_graph_data(root.action, labels, xs, ys))
    button.grid(row=row+1, column=column+1, padx=5, pady=5)


# Function allowing to extract the data from a graph
def extract_graph_data(action, labels, xs, ys):
    # Create folders if they don't exist
    if not os.path.exists("export"):
        os.mkdir("export")
    if not os.path.exists(f"export/{action.folder_name}"):
        os.mkdir("export/" + action.folder_name)
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(f"export/{action.folder_name}/{datetime.now().strftime('%y%m%d_%H%M%S')}.xlsx",
                            engine="xlsxwriter")

    for i in range(len(labels)):
        df = pd.DataFrame({
            labels[i][0]: xs[i],
            labels[i][1]: ys[i]
        })
        df.to_excel(writer, sheet_name=(str(i+1) + " " + labels[i][2]), index=False)
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


# Build a graph with the x and y values
def build_box_plot(root, label,  x, ys, row, column, title, x_label, y_label):
    fig = Figure(figsize=(6, 4), dpi=100)
    a = fig.add_subplot(111)

    # Setting the title and axis labels
    a.set_title(title)
    a.set_xlabel(x_label)
    a.set_ylabel(y_label)

    # Plotting the box plot
    a.boxplot(ys, labels=x)

    # Creation of the canvas
    canvas = FigureCanvasTkAgg(fig, root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=row, column=column, columnspan=2, padx=5, pady=5)

    # Creation of the toolbar, displayed under the graph
    toolbar_frame = Frame(root)
    toolbar_frame.grid(row=row+1, column=column)
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()
    canvas.get_tk_widget().grid(row=row, column=column, columnspan=2,  padx=5, pady=5)

    # Button to extract data
    button = ttk.Button(root, text="Export data",
                        command=lambda: extract_full_data(root, [label], [x], [ys]))
    button.grid(row=row + 1, column=column + 1, padx=5, pady=5)


# Button for extracting data from a graph
def extract_full_data_button(root, labels, xs, yss, row, column):
    # Button to extract data
    button = ttk.Button(root, text="Export All Data",
                        command=lambda: extract_full_data(root, labels, xs, yss))
    button.grid(row=row, column=column, padx=5, pady=5)


# Extract all the number of iterations in the runs
def extract_full_data(root, labels, xs, yss):
    # Create folders if they don't exist
    if not os.path.exists("export"):
        os.mkdir("export")
    if not os.path.exists(f"export/{root.action.folder_name}"):
        os.mkdir(f"export/{root.action.folder_name}")
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(f"export/{root.action.folder_name}/{datetime.now().strftime('%y%m%d_%H%M%S')}_"
                            f"{root.fitness_function_name.get()}.xlsx",
                            engine="xlsxwriter")
    for i in range(len(labels)):
        df = pd.DataFrame()
        for j in range(len(xs[i])):
            df[str(xs[i][j])] = yss[i][j]
        df.to_excel(writer,  sheet_name=(str(i+1) + " " + labels[i]), index=True)
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


# Return an array with the default value of the parameters of the element
def default_parameters(element, size):
    parameter_values = []
    for parameter in element.parameters:
        # If we use the interface, create an IntVar
        if sys.argv[1] == "0":
            parameter_value = IntVar()
            # Get default value, which can be a function of the problem size
            parameter_value.set(update_parameter(parameter.default_value, size))
            parameter_values.append(parameter_value)
        # If running a script, just use normal integers
        else:
            parameter_values.append(update_parameter(parameter.default_value, size))
    return parameter_values


# Function running n times an algorithm on a fitness function using parallel programming
def run_parallel(iterations, size, evolutionary_algorithm, evolutionary_parameter_values,
                 fitness_function, fitness_parameter_values, cores):
    pool = multiprocessing.Pool(cores)
    results = pool.map(functools.partial(solve_partial,
                       size, evolutionary_algorithm, evolutionary_parameter_values,
                       fitness_function, fitness_parameter_values), range(iterations))
    pool.close()
    return np.array(results)


# Function solving an evolutionary algorithm on a fitness function and
# returning the number of call to the fitness function
def solve_partial(size, evolutionary_algorithm, evolutionary_parameter_values,
                  fitness_function, fitness_parameter_values, i):
    _, iterations, _, _, _ = evolutionary_algorithm.solve_fitness(evolutionary_parameter_values, size, fitness_function,
                                                                  fitness_parameter_values, False)
    return iterations


# Function allowing parallel programming for the cGA while returning its final probabilities
def run_parallel_cga(iterations, size, max_iter, evolutionary_algorithm, evolutionary_parameter_values,
                     fitness_function, fitness_parameter_values, cores):
    pool = multiprocessing.Pool(cores)
    results = pool.map(functools.partial(solve_partial_cga,
                       size, max_iter, evolutionary_algorithm, evolutionary_parameter_values,
                       fitness_function, fitness_parameter_values), range(iterations))
    pool.close()
    return np.array(results)


def solve_partial_cga(size, max_iter,  evolutionary_algorithm, evolutionary_parameter_values,
                      fitness_function, fitness_parameter_values, i):
    _, iterations, _, _, ys = evolutionary_algorithm.solve_fitness(evolutionary_parameter_values, size,
                                                                   fitness_function, fitness_parameter_values,
                                                                   False, max_iter, True)

    return np.concatenate([[iterations], ys])


def run_parallel_sat(evolutionary_algorithm, evolutionary_parameter_values,
                     sat_problems, cores):
    pool = multiprocessing.Pool(cores)
    results = pool.map(functools.partial(solve_partial_sat,
                       evolutionary_algorithm, evolutionary_parameter_values), sat_problems)
    pool.close()
    return np.array(results)


# Function solving an evolutionary algorithm on a fitness function and
# returning the number of call to the fitness function
def solve_partial_sat(evolutionary_algorithm, evolutionary_parameter_values, sat_problem):
    _, iterations, _, _, _ = evolutionary_algorithm.solve_SAT(evolutionary_parameter_values, sat_problem, False)
    return sat_problem.name, iterations




