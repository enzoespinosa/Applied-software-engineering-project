import itertools
import random
import HopfieldNetwork 
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def dynamics(state, weights, max_iter=20):
    """
    The function takes in a state, a weight matrix, the maximum number of iterations and the number of iterations before we consider there is convergence as input and returns the
    dynamic history of the state. It also prints in how many steps the convergence has been achieved.

    Parameters
    ----------
    state
        the initial state of the network
    weights
        the weight matrix
    max_iter
        the maximum number of iterations to run the dynamics for, set by default to 20
    convergen_num_iter
        The number of iterations that the network has to be stable for before we consider it converged, set by default to 10000
    Returns
    -------
    The dynamic history of the network.
    """
    prev_state = state.copy()  
    dynamic_history = np.array([state])
    for i in range(1,max_iter+1):
        next_state = HopfieldNetwork.update(prev_state,weights)
        if np.array_equal(next_state, prev_state):
            print("Convergence achieved in {} iterations".format(i))
            dynamic_history = np.append(dynamic_history, [next_state], axis=0)
            prev_state = next_state
            break
        dynamic_history = np.append(dynamic_history, [next_state], axis=0)
        prev_state = next_state
    
    return dynamic_history