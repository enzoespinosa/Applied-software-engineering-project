import itertools
import random

import numpy as np
import cython
import HopfieldNetwork

def dynamics_async(state, weights, max_iter, convergence_num_iter = 10000):
    """
    The function takes in the initial state, the weights, the maximum number of iterations, and the
    number of iterations required for convergence. It then runs the asynchronous update function until
    convergence is achieved or the maximum number of iterations is reached.
    
    Parameters
    ----------
    state
        the initial state of the system
    weights
        the weight matrix
    max_iter
        the maximum number of iterations to run the dynamics for
    convergence_num_iter
        The number of iterations that the system has to be in a stable state for it to be considered
    converged.
    
    Returns
    -------
    The history of the state of the system.
    """
    prev_state = state.copy()
    counter = 0
    dynamic_async_history = np.array([state])
    for i in range(max_iter):
        next_state = HopfieldNetwork.update_async(prev_state,weights)
        counter = counter+1 if np.array_equal(next_state, prev_state) else 0
        if counter == convergence_num_iter :
            print ("Convergence has been achieved in :", i+1, " steps ")
            dynamic_async_history = np.append(dynamic_async_history, [next_state], axis=0)
            break
        if i%1000 == 0 :
            print("Iteration: ", i)
            dynamic_async_history = np.append(dynamic_async_history, [next_state], axis=0)
        
        prev_state = next_state
    return dynamic_async_history