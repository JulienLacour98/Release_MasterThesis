import random
import numpy as np


class BitString:

    # Generates a random bit string of length "size"
    def __init__(self, size=0):
        self.string = ""
        for i in range(size):
            self.string += str(random.randint(0, 1))

    # Flip every bit with a probability of p
    def create_offspring_p(self, p):
        # Creation of the new bit string by flipping bits from the previous bit string
        new_string = ""
        for bit in self.string:
            if random.random() < p:
                new_string += str(1 - int(bit))
            else:
                new_string += bit
        new_bit_string = BitString()
        new_bit_string.string = new_string
        return new_bit_string

    # Flip s bits uniformly
    def create_offspring_s(self, s):
        n = len(self.string)
        # Generates an array of "s" unique random integer in [0, n-1]
        indexes = random.sample(list(np.arange(0, n)), s)
        new_string = ""
        for i in range(n):
            # Switch the randomly chosen bits
            if i in indexes:
                new_string += str(1 - int(self.string[i]))
            else:
                new_string += self.string[i]
        new_bit_string = BitString()
        new_bit_string.string = new_string
        return new_bit_string

    # Change the first 0 in the bitstring to 1
    # Used in order to raise the norm by 1 for creating the graph of a fitnesss function
    def add_one_one(self):
        new_string = ""
        i = 0
        # Keep the same prefix of the bit string
        while i < len(self.string) and self.string[i] == "1":
            new_string += self.string[i]
            i += 1
        # Switch the first 0 to 1
        if i < len(self.string):
            new_string += "1"
            i += 1
        else:
            raise Exception("Can't add any one to the only ones string")
        # Keep the same suffix
        for j in range(i, len(self.string)):
            new_string += self.string[j]
        self.string = new_string


# Create bit string with only 0s
def only_zeros(size):
    bit_string = BitString()
    bit_string.string = "0" * size
    return bit_string


# Create bit string with only 1s
def only_ones(size):
    bit_string = BitString()
    bit_string.string = "1" * size
    return bit_string
