import numpy as np
from numba.types import int64

# Define the species variable
species = 3

# Calculate the number of int64 values needed
num_int64 = species * 2

# Generate the tuple of int64 dynamically
int_tuple = tuple([int64] * num_int64)

print(int_tuple)

int_tuple = tuple
