import numpy as np
import random


np.random.seed(0)  # Set the seed for reproducibility
arr = np.random.randint(0, 10, 10)  # Generate an array of 10 random numbers
arr = arr.reshape(-1, 5)

norm_arr = np.linalg.norm(arr)

matrix_by_norm = arr / norm_arr
print(arr)

print(
    f"The norm of the array is: \n {norm_arr} \n and matrix/norm = \n {arr/norm_arr} \n norm of norm = \n {np.linalg.norm(matrix_by_norm)}"
)
