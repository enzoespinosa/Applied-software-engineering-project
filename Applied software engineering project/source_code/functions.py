import itertools
import random
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import HopfieldNetwork as hn

def random_choice():
    """
    This function returns a random choice of either -1 or 1.

    Parameters
    ----------
    There are no parameters.

    Returns
    -------
    A random choice of either -1 or 1.
    """
    rand = np.random.choice([-1,1])
    return rand

def create_vector(n):
    """
    This function creates a vector of length n, where each element is either a 1 or a -1, with equal
    probability, thanks to random_choice function.

    Parameters
    ----------
    n
        the number of rows of the vector of 1s and -1s
    
    Returns
    -------
    A vector of length n with random values of 1 or -1.
    """
    matrix = np.ones(n)
    for i in range(0,n-1):
        matrix[i] = random_choice()
    return matrix



def generate_patterns(m=3,n=50):
    """
    It generates a matrix of m rows and n columns, where each row is a random binary vector of length n.
    m is the number of patterns (set by default to 3) and n the number of neurons (set by default to 50).

    Parameters
    ----------
    m
        the number of patterns
    n
        the number of neurons for a single pattern

    Returns
    -------
    A matrix of m patterns (rows) of n neurons (columns)
    """
    patterns = np.zeros(0)
    for i in range (m):
        patterns = np.append(patterns,create_vector(n))
    return np.reshape(patterns, (m,n))


def perturb_pattern(pattern, num_perturb=2):
    """
    The function takes a pattern and returns a perturbed version of the pattern, with a decided number or perturbations.

    Parameters
    ----------
    pattern
        a vector of neurons with values from {-1,1}
    num_perturb
        int value, refers to the number of perturbations in pattern (set by default to 3)

    Returns
    -------
    The perturbed pattern (same size as pattern, but with num_perturb neuron values inverted)
    """
    new_pattern=pattern.copy()
    pattern_max_idx=len(pattern)-1
    #itertools is faster
    for _ in itertools.repeat(None, num_perturb):
       idx = random.randint(0,pattern_max_idx)
       new_pattern[idx] = -new_pattern[idx]
       
    return new_pattern	 

def pattern_match(memorized_patterns, pattern):
    """
    If the pattern is in the list of memorized patterns, return the index of the pattern in the list.
    Otherwise, return None.
    
    Parameters
    ----------
    memorized_patterns
        a list of patterns that have been memorized
    pattern
        The pattern to be matched

    Returns
    -------
    The index of the pattern in the memorized_patterns list.
    """
    try:
        index = memorized_patterns.index(pattern)
        return index
    except ValueError:
        return None    


def hebbian_weights(patterns):
    """
    The function takes a list of patterns and returns the Hebbian weights matrix.

    Parameters
    ----------
    patterns
        a list of patterns, each of which is a list of 1s and -1s

    Return
    ------
    The weights matrix
    """
    n = len(patterns[0])
    m = len(patterns)
    weights = np.zeros((n,n))
    for el in patterns:
        weights += np.outer(el,el)
    w=weights/m  
    np.fill_diagonal(w,0)          
    return w


def update(state,weights):
    """
    It takes a state and a weight matrix and returns the next state.

    Parameters
    ----------
    state
        the current state of the network
    weights
        the weight matrix
    
    Returns
    -------
    The next state of the network.
    """
    next_pattern = np.array([1 if elem >= 0 else -1 for elem in np.dot(weights,state)])
    return next_pattern
    


def update_async(state,weights):
    """
    It takes a state and a set of weights and returns a new state where one of the spins has been
    flipped.

    Parameters
    ----------
    state
        the current state of the network
    weights
        a matrix of weights, where each row is a neuron and each column is a neuron's input
    
    Returns
    -------
    The new state of the system.
    """
    idx = random.randint(0,len(state)-1)
    new_state = state.copy()
    new_state[idx] = 1 if np.dot(weights,state)[idx] >=0 else -1
    return new_state


def pattern_choice(patterns):
    """
    It takes a list of patterns and returns one of them at random.

    Parameters
    ----------
    patterns
        a list of patterns to choose from

    Returns
    -------
    A random pattern from the list of patterns.
    """
    return patterns[random.randint(0,len(patterns)-1)]


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
        next_state = update(prev_state,weights)
        if np.array_equal(next_state, prev_state):
            print("Convergence achieved in {} iterations".format(i))
            dynamic_history = np.append(dynamic_history, [next_state], axis=0)
            prev_state = next_state
            break
        dynamic_history = np.append(dynamic_history, [next_state], axis=0)
        prev_state = next_state
    
    return dynamic_history

   


# def dynamics_async(state, weights, max_iter, convergence_num_iter = 10000):
#     """
#     The function takes in the initial state, the weights, the maximum number of iterations, and the
#     number of iterations required for convergence. It then runs the asynchronous update function until
#     convergence is achieved or the maximum number of iterations is reached.
    
#     Parameters
#     ----------
#     state
#         the initial state of the system
#     weights
#         the weight matrix
#     max_iter
#         the maximum number of iterations to run the dynamics for
#     convergence_num_iter
#         The number of iterations that the system has to be in a stable state for it to be considered
#     converged.
    
#     Returns
#     -------
#     The history of the state of the system.
#     """
#     prev_state = state.copy()
#     counter = 0
#     dynamic_async_history = np.array([state])
#     for i in range(max_iter):
#         next_state = update_async(prev_state,weights)
#         counter = counter+1 if np.array_equal(next_state, prev_state) else 0
#         if counter == convergence_num_iter :
#             print ("Convergence has been achieved in :", i+1, " steps ")
#             dynamic_async_history = np.append(dynamic_async_history, [next_state], axis=0)
#             break
#         if i%1000 == 0 :
#             print("Iteration: ", i)
#             dynamic_async_history = np.append(dynamic_async_history, [next_state], axis=0)
        
#         prev_state = next_state
#     return dynamic_async_history


def storkey_weights(patterns):
    """
    The function takes a list of patterns and returns the Storkey weights matrix.

    Parameters
    ----------
    patterns
        a list of patterns, each of which is a list of 1s and -1s

    Return
    ------
    The weights matrix
    """
    
    N = np.size(patterns[0])
    W, H = np.zeros((N, N)), np.zeros((N, N))
    previous_storkey_weights = np.zeros((N, N)) 
    for mu in range(np.shape(patterns)[0]):
        matrix = np.tile(patterns[mu], (N, 1)).T    
        diag_matrix = matrix.copy()
        np.fill_diagonal(diag_matrix, 0)
        diag_storkey = previous_storkey_weights.copy()
        np.fill_diagonal(diag_storkey, 0)
        matrix_of_patt_diag = np.diag(np.diag(matrix))
        H = np.matmul(diag_storkey, diag_matrix)
        
        H_diag_patterns = np.matmul(H.copy(), matrix_of_patt_diag)
        matrix_patt_diag = np.matmul(matrix, matrix_of_patt_diag)
        W = previous_storkey_weights + (1/N)*(matrix_patt_diag - H_diag_patterns - H_diag_patterns.T)
        previous_storkey_weights = W.copy()
    return W


def energy(state, weights):
    """
    It takes a state and a weight matrix and returns the energy of that state.

    Parameters
    ----------
    state
        a vector of length N, where N is the number of neurons in the network
    weights
        the weight matrix

    Returns
    -------
    The energy of the state.
    """
    return -0.5*np.dot(state, np.dot(weights,state.transpose()))


def energy_evolution(patterns, weights):
    """
    It takes a list of patterns and a weight matrix and returns the energy of each pattern.

    Parameters
    ----------
    patterns
        a list of patterns, each of which is a numpy array of shape (N,1)
    weights
        the weight matrix

    Returns
    -------
    A list of energies, one for each pattern.
    """
    energy_history = np.array([])
    for pattern in patterns:
         energy_history = np.append(energy_history, energy(pattern,weights))
    return energy_history

def energy_is_not_increasing(energy_history):
    """
    It takes a list of energies and returns True if the energy is not increasing, False otherwise.

    Parameters
    ----------
    energy_history
        a list of energies

    Returns
    -------
    True if the energy is not increasing, False otherwise.
    """
    return np.all(np.diff(energy_history) <= 0)

def energy_plot(energy_history,is_async=False):
    """
    It takes a list of energies and plots them.
    
    Parameters
    ----------
    energy_history
        a list of energies
    """
    plt.figure()
    x_axis = np.arange(0,len(energy_history),1)
    y_axis = energy_history
    plt.title("Energy evolution")
    if is_async:
        plt.xlabel("Iterations in thousands")
    else:
        plt.xlabel("Iterations")
    plt.ylabel("Energy")
    plt.plot(x_axis, y_axis, color ="red")
    plt.show()



def checkerboard_visualization(size):
    """
    It creates a checkerboard pattern of alternating 1s and -1s, with a 5x5 checkers of 1s and -1s, with an overall size of size x size .

    Parameters
    ----------
    size
        the size of the checkerboard

    Returns
    -------
    A checkerboard of size (size,size).
    """
    checkerboard = np.zeros((size,size))
    for i in range(0,size,5):
        for j in range(0,size,5):
            checkerboard[i:i+5,j:j+5] = -1 if (i+j)%10==0 else 1
    return checkerboard.flatten()


def replace_random_pattern(patterns, checkerboard):
    """
    Replace a random pattern in the list of patterns with the checkerboard pattern.

    Parameters
    ----------
    patterns
        a list of patterns to be used in the simulation
    checkerboard
        a 2D array of 1s and -1s

    Returns
    -------
    The patterns list with the checkerboard pattern replacing a random pattern in the list.
    """
    idx = random.randint(0,len(patterns)-1)
    patterns[idx] = checkerboard
    return patterns
    

def save_video(state_list, out_path):
    """
    It takes a list of 2D numpy arrays and saves them as a gif.

    Parameters
    ----------
    state_list
        a list of numpy arrays, each of which is a state of the game
    out_path
        the path to save the video

    Returns
    -------
    Nothing

    Visualization
    -------------
    Animation of perturbed checkborad pattern converging to stable checkboard pattern.
    """
    fig = plt.figure()
    plt.title("Convergence experiment")
    ims = []
    for state in state_list:
        im = plt.imshow(state, cmap='gray', aspect='equal',origin= 'lower', extent=None, interpolation_stage=None, filternorm=True,
        filterrad=4.0, resample=None, url=None, data=None)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=300, blit=True,repeat_delay=1000)
    ani.save(out_path, writer='pillow', fps=30)
    plt.show()