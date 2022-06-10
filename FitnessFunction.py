from Parameter import *
from Constraint import *


class FitnessFunction:

    def __init__(self, name, parameters, size_constraint, function, function_maximum):
        self.name = name
        self.parameters = parameters
        self.size_constraint = size_constraint
        self.function = function
        self.function_maximum = function_maximum
        # List with all the fitness functions
        fitness_functions.append(self)
        # List with all the fitness function names used for dropdown menus
        fitness_function_names.append(self.name)

    # Return the image of the fitness function for the bit string
    def result(self, parameters, bit_string):
        return self.function(parameters, bit_string.string)

    # Return the maximum of the fitness function for a selected size
    def maximum(self, parameters, size):
        return self.function_maximum(parameters, size)


# Definition of the OneMax function
def one_max(parameters, bit_string):
    return bit_string.count('1')


# Function returning the maximum of the OneMax function
def one_max_maximum(parameters, n):
    return n


# Definition of the Jump function
def jump_m(parameters, bit_string):
    n = len(bit_string)
    m = parameters[0]
    norm = bit_string.count('1')
    if norm <= n - m or norm == n:
        return m + norm
    else:
        return n - norm


# Function returning the maximum of the Jump function
def jump_m_maximum(parameters, n):
    return parameters[0] + n


# Definition of the JumpOff function
def jump_off_m_c(parameters, bit_string):
    n = len(bit_string)
    m = parameters[0]
    c = parameters[1]
    norm = bit_string.count('1')
    if norm <= n - m - c or norm >= n - c:
        return m + norm
    else:
        return n - norm - c


# Function returning the maximum of the JumpOff function
def jump_off_m_c_maximum(parameters, n):
    return parameters[0] + n


# Definition of the JumpOffset function
def jump_offset_m(parameters, bit_string):
    n = len(bit_string)
    m = parameters[0]
    norm = bit_string.count('1')
    if norm <= 3 * n/4 or norm >= 3 * n/4 + m:
        return m + norm
    else:
        return 3 * n/4 + m - norm


# Function returning the maximum of the JumpOffset function
def jump_offset_m_maximum(parameters, n):
    return parameters[0] + n


# Definition of the JumpOffsetSpike function
def jump_offset_spike_m(parameters, bit_string):
    n = len(bit_string)
    m = parameters[0]
    norm = bit_string.count('1')
    if norm <= 3 * n/4 or norm >= 3 * n/4 + m:
        return m + norm
    elif norm == 3*n/4 + m/2:
        return n + m + 1
    else:
        return 3*n/4 + m - norm


# Function returning the maximum of the JumpOffsetSpike function
def jump_offset_spike_m_maximum(parameters, n):
    return n + parameters[0] + 1


# Definition of the Cliff function
def cliff_d(parameters, bit_string):
    n = len(bit_string)
    d = parameters[0]
    norm = bit_string.count('1')
    if norm <= n - d:
        return norm
    else:
        return norm - d + 1/2


# Function returning the maximum of the Cliff function
def cliff_d_maximum(parameters, n):
    d = parameters[0]
    return n - d + 1/2


# Definition of the Hurdle function
def hurdle_w(parameters, bit_string):
    w = parameters[0]
    z = bit_string.count('0')
    r = z % w
    return - math.ceil(z/w) - r/w


# Function returning the maximum of the Hurdle function
def hurdle_w_maximum(parameters, n):
    return 0


# Definition of NeedGlobalMut
def need_global_mut(parameters, bit_string):
    n = len(bit_string)
    block_size = math.ceil(math.pow(n, 1/4))
    number_of_blocks = math.ceil(math.sqrt(n)/3)
    m = block_size * number_of_blocks
    prefix_validity, prefix_value = prefix(n, m, bit_string)
    suffix_validity, suffix_value = suffix(n, block_size, number_of_blocks, m, bit_string)
    if prefix_validity and suffix_validity:
        if prefix_value <= 9*(n-m)/10:
            return int(math.pow(n, 2) * suffix_value + prefix_value)
        else:
            return int(math.pow(n, 2) * number_of_blocks + prefix_value + suffix_value - n - 1)
    else:
        return -bit_string.count("1")


# Function returning the maximum of the NeedGlobalMut function
def need_global_mut_maximum(parameters, n):
    block_size = math.ceil(math.pow(n, 1/4))
    number_of_blocks = math.ceil(math.sqrt(n) / 3)
    m = block_size * number_of_blocks
    max_prefix = math.floor(9*(n-m)/10)
    max_suffix = number_of_blocks
    return int(math.pow(n, 2) * max_suffix + max_prefix)


# Auxiliary functions for NeedGlobalMut
# Function returning if the prefix is valid, and the correct value if it is the case
def prefix(n, m, bit_string):
    prefix_size = n - m
    idx = 0
    while idx < prefix_size and bit_string[idx] == "1":
        idx += 1
    value = idx
    while idx < prefix_size and bit_string[idx] == "0":
        idx += 1
    return idx == prefix_size, value


# Function returning if the suffix is valid, and the correct value if it is the case
def suffix(n, block_size, number_of_blocks, m, bit_string):
    prefix_size = n - m
    active_blocks = ""
    for i in range(number_of_blocks):
        count_ones = 0
        for j in range(prefix_size + block_size * i, prefix_size + block_size * (i + 1)):
            if bit_string[j] == "1":
                count_ones += 1
        if count_ones == 0:
            active_blocks += "0"
        elif count_ones == 2:
            active_blocks += "1"
        else:
            return False, -1
    idx = 0
    while idx < number_of_blocks and active_blocks[idx] == "1":
        idx += 1
    value = idx
    while idx < number_of_blocks and active_blocks[idx] == "0":
        idx += 1
    return idx == number_of_blocks, value


# List containing every fitness functions
fitness_functions = []
fitness_function_names = []

# Creation of OneMax
OneMax = FitnessFunction("OneMax", [], INT, one_max, one_max_maximum)

# Creation of Jump
gap_m = Parameter("m", "integer", 4, 2, "size", INT)
JumpM = FitnessFunction("Jump_m", [gap_m], INT, jump_m, jump_m_maximum)

# Creation of JumpOff
# ps: very similar to JumpOffset but deciding where the jump is
gap_m = Parameter("m", "integer", 4, 2, "size", INT)
length_c = Parameter("c", "integer", 5, 0, "size", INT)
JumpOffMC = FitnessFunction("JumpOff_m_c", [gap_m, length_c], INT, jump_off_m_c, jump_off_m_c_maximum)

# Creation of JumpOffset
gap_m = Parameter("m", "integer", 4, 2, "size/4", INT)
JumpOffsetM = FitnessFunction("JumpOffset_m", [gap_m], M4, jump_offset_m, jump_offset_m_maximum)

# Creation of JumpOffsetSpike
gap_m = Parameter("m", "integer", 4, 2, "size/4", M2)
JumpOffsetSpikeM = FitnessFunction("JumpOffsetSpike_m", [gap_m], M4, jump_offset_spike_m, jump_offset_spike_m_maximum)

# Creation of Cliff
gap_d = Parameter("d", "integer", 4, 1, "size", INT)
CliffD = FitnessFunction("Cliff_d", [gap_d], INT, cliff_d, cliff_d_maximum)

# Creation of Hurdle
param_w = Parameter("w", "integer", 4, 2, "size", INT)
HurdleW = FitnessFunction("Hurdle_w", [param_w], INT, hurdle_w, hurdle_w_maximum)

# Creation of NeedGlobalMut
NeedGlobalMutM = FitnessFunction("NeedGlobalMut", [], INT, need_global_mut, need_global_mut_maximum)
