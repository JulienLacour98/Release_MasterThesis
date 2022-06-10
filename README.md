# MasterThesis
Implementation and Evaluation of Strategies for Escaping Local Optima with Evolutionary Algorithms

# Code
The code can also be found with the following url: https://github.com/JulienLacour98/Release_MasterThesis.git

# Libraries
The following libraries need to be imported:
- matplotlib
- numpy
- openpyxl
- pandas
- XlsxWriter

# Exports / Imports
- All exported data will be found in the folder "export" and sorted in an action folder
  - Data exported from a script "i" will be in the folder "export/script_i" where the name file is the parameters used. Using the same parameters again will erase the previous run. 
  - Data exported from the interface within an "action" will be under the folder "export/interface/action" and will have the data as a name file.

# Running the framework
- In order to run the framework use the following command in the terminal: "python3 Main.py [script] [params]"
  - For [script] = 0, the interface will be launched. [params] is then not needed
  - For [script] in {1,2,3,4}, the "Main.py" will start a script. The scripts are defined in "Script.py".
    - In that case, [params] is not empty, and the parameters necessary for the script needs to be defined.
    - For that, look at the "Main.py" and "Script.py" files to see what input is needed.
    - Example of command line for starting the scripts are found in the last line of each file in the "script" folder.