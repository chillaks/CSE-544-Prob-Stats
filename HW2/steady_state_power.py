import numpy as np

# returns the steady state probability matrix
def steady_state_power(transition_matrix):
    # set k to a high value, k >> 1
    k = 100
    a = np.linalg.matrix_power(transition_matrix, k)
    return a


transition_matrix = [[0.9, 0.0, 0.1, 0.0],
                     [0.8, 0.0, 0.2, 0.0],
                     [0.0, 0.5, 0.0, 0.5],
                     [0.0, 0.1, 0.0, 0.9]]
a = steady_state_power(transition_matrix)
print("Steady_State: Power iteration >> " + str(a[0,:]))