import functools


class Constraint:

    def __init__(self, name, description, condition):
        self.name = name
        self.description = description
        self.condition = condition

    # Return True if the condition is met
    def check_condition(self, element):
        return self.condition(element)


# Return True if the element is an integer
def is_integer(element):
    return str(element).isnumeric()


# Return True if the element is a multiple of k
def multiple_of_k(k, element):
    return int(element) % k == 0


# Constraint that the element needs to be an integer
INT = Constraint("INT", "This has to be an integer", is_integer)

# Constraint that the element needs to be a multiple of 2
M2 = Constraint("M2", "This integer has to be a multiple of 2", functools.partial(multiple_of_k, 2))

# Constraint that the element needs to be a multiple of 4
M4 = Constraint("M4", "This integer has to be a multiple of 4", functools.partial(multiple_of_k, 4))
