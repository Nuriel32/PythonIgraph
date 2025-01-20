import numpy as np
from collections import deque


def find_minimum_distance_with_steps(A, B, C, D):
    # Function to multiply C by one of A, A^-1, B, B^-1 and return the new state with the corresponding operation
    def multiply_and_get_next_states(C):
        next_states = []
        operations = ['A', 'A^-1', 'B', 'B^-1']
        for i, M in enumerate([A, A.T, B, B.T]):
            next_states.append((np.dot(M, C), operations[i]))  # M * C
            next_states.append((np.dot(C, M), 'C' + operations[i]))  # C * M
        return next_states

    # BFS setup
    queue = deque([(C, 0, "")])  # Queue contains tuples of (matrix, distance, operation string)
    visited = set()
    visited.add(tuple(C.flatten()))

    while queue:
        current_matrix, dist, operation_str = queue.popleft()

        # If the current matrix equals D, return the distance and operation string
        if np.array_equal(current_matrix, D):
            return dist, operation_str

        # Generate next possible states
        for next_matrix, operation in multiply_and_get_next_states(current_matrix):
            flattened_next_matrix = tuple(next_matrix.flatten())
            if flattened_next_matrix not in visited:
                visited.add(flattened_next_matrix)
                queue.append((next_matrix, dist + 1, operation_str + " * " + operation))

    return -1, ""  # If no solution is found



#matrix calc
def perform_multiplication_sequence(A, B, C, sequence):
    # Split the sequence into individual operations
    operations = sequence.split(' * ')

    # Start with the initial matrix C
    current_matrix = C

    # Perform the operations in sequence
    for operation in operations:
        if operation == 'A':
            current_matrix = np.dot(A, current_matrix)
        elif operation == 'A^-1':
            current_matrix = np.dot(np.linalg.inv(A), current_matrix)
        elif operation == 'B':
            current_matrix = np.dot(B, current_matrix)
        elif operation == 'B^-1':
            current_matrix = np.dot(np.linalg.inv(B), current_matrix)
        elif operation == 'CA':
            current_matrix = np.dot(current_matrix, A)
        elif operation == 'CA^-1':
            current_matrix = np.dot(current_matrix, np.linalg.inv(A))
        elif operation == 'CB':
            current_matrix = np.dot(current_matrix, B)
        elif operation == 'CB^-1':
            current_matrix = np.dot(current_matrix, np.linalg.inv(B))

    return current_matrix
A = np.array([
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 0]
])

# Define matrix B
B = np.array([
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0]
])

# Define matrix C
C = np.array([
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0]
])

# Define matrix D
D = np.array([
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 0]
])
# Calculate minimum distance and get the operations
distance, operations = find_minimum_distance_with_steps(A, B, C, D)
print(f"Minimum distance: {distance}")  # Should print 7
print(f"Operations: {operations.strip(' * ')}")  # Should print the correct sequence of 7 operations

sequence = operations.strip(' * ')

# Calculate the resulting matrix
result_matrix = perform_multiplication_sequence(A, B, C, sequence)

print("Resulting Matrix:")
print(result_matrix)


# Compute A * A
AA = np.dot(A, A)

# Compute A^-1 (inverse of A)
A_inv = np.linalg.inv(A)

# Compute CA^-1
CA_inv = np.dot(C, A_inv)

# Compute CA^-1 * C
C_A_inv_C = np.dot(CA_inv, C)

# Compute full expression A * A * CA^-1 * C * A^-1 * B * C * B * C * A^-1
result = np.dot(AA, np.dot(C_A_inv_C, np.dot(A_inv, np.dot(B, np.dot(C, np.dot(B, np.dot(C, A_inv)))))))

print("this is result")
print(result)