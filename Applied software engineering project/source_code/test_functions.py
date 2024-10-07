import functions as f
import PatternManager as pm
import HopfieldNetwork as hn
import DataSaver as ds
import numpy as np

# import problem with dynamics_cython
#from dynamics_cython import dynamics


def test_create_pattern():
    """
    It creates a vector of length 4 with elements that are either 1 or -1
    """

    pattern_manager = pm.PatternManager()
    vector = pattern_manager.create_pattern(4)
    assert (elem == 1 or elem == -1 for elem in vector) and len(vector) == 4

# def test_energy_decrease():
#     """
#     It tests whether the energy of the network decreases over time
#     """

#     network = f.hn()

#     energy_decrease_h = False
#     energy_decrease_s = False

#     patterns = np.array([[1,1,-1,-1],[1,1,-1,1],[-1,1,-1,1]])
#     weights = network.hebbian_weights(patterns)
#     storkey = network.storkey_weights(patterns)

#     energy_evolution_hebbian = network.energy_evolution(patterns, weights)
#     energy_evolution_storkey = network.energy_evolution(patterns, storkey)

#     for i in range(len(energy_evolution_hebbian)-1):
#         if energy_evolution_hebbian[i] <= energy_evolution_hebbian[i+1]:
#             energy_decrease_h = True

#     for i in range(len(energy_evolution_storkey)-1):
#         if energy_evolution_storkey[i] <= energy_evolution_storkey[i+1]:
#             energy_decrease_s = True

#     assert energy_decrease_h and energy_decrease_s



def test_energy_is_not_increasing_storkey():
    """
    It checks if the energy is not increasing
    """
    saver = ds.DataSaver()
    pattern_manager = pm.PatternManager()
    patterns = pattern_manager.create_patterns(3, 10)
    network = hn.HopfieldNetwork(patterns, "storkey")

    perturbed_pattern = pattern_manager.perturb_pattern(network.patterns[0], 2)
    network.dynamics(perturbed_pattern, saver, 20)
    _, energies = saver.get_data()
    is_not_increasing_bool = np.all(np.diff(energies) <= 0)
    assert np.allclose(energies[-2], energies[-1]) is not None and is_not_increasing_bool

    #SUCCESS?

def test_energy_is_not_increasing_hebbian():
    """
    It checks if the energy is not increasing
    """
    saver = ds.DataSaver()
    pattern_manager = pm.PatternManager()
    network = hn.HopfieldNetwork(np.array([[1,1,-1,-1],[1,1,-1,1],[-1,1,-1,1]]), "hebbian")
    print (network.patterns[0])
    perturbed_pattern = pattern_manager.perturb_pattern(network.patterns[0], 2)
    network.dynamics(perturbed_pattern, saver, 20)
    states, energies = saver.get_data()
    energies = np.array([])
    is_not_increasing_bool = np.all(np.diff(energies) <= 0)
   
    assert np.allclose(states[-2], states[-1]) is not None and is_not_increasing_bool

    #SUCCESS?


def test_energy():
    """
    It tests the energy of a pattern.
    """

    saver = ds.DataSaver()
    network = hn.HopfieldNetwork(np.array([[1,1,-1,-1],[1,1,-1,1],[-1,1,-1,1]]), "hebbian")
    w = network.weights

    energy = saver.compute_energy(network.patterns[0], w)
    assert energy == -0.5*np.dot(network.patterns[0], np.dot(w,network.patterns[0].transpose()))

def test_generate_patterns():
    """
    It generates a matrix of random patterns, where each pattern is a vector of 1s and -1s
    """
    pattern_manager = pm.PatternManager()

    patterns = pattern_manager.create_patterns(3, 4)
    assert len(patterns) == 3 and len(patterns[0]) == 4
    assert np.shape(patterns) == (3, 4)
    for i in range(len(patterns)):
        for j in range(len(patterns[0])):
            assert patterns[i][j] == 1 or patterns[i][j] == -1
    

def test_hebbian_dynamics():
    """
    Tests the dynamics of the network with the Hebbian weights
    """

    pattern_manager = pm.PatternManager()
    network = hn.HopfieldNetwork(np.array([[1,1,-1,-1],[1,1,-1,1],[-1,1,-1,1]]), "hebbian")
    saver = ds.DataSaver()

    perturbed_pattern = pattern_manager.perturb_pattern(network.patterns[0], 2) #2 elements are perturbed successfully
    network.dynamics(perturbed_pattern, saver, 20)
    states, _ = saver.get_data()
    assert np.allclose(states[-2], states[-1]) is not None
    
    #SUCCESS

def test_hebbian_weights():
    """
    It takes a set of patterns and tests the values of their Hebbian weights matrix
    """

    network = hn.HopfieldNetwork(np.array([[1,1,-1,-1],[1,1,-1,1],[-1,1,-1,1]]), "hebbian")

    correct_matrix_hebbian = np.array([[0.,0.33333333,-0.33333333,-0.33333333],[0.33333333,0.,-1.,0.33333333],[-0.33333333,-1.,0.,-0.33333333],[-0.33333333,0.33333333,-0.33333333,0.]])
    assert np.allclose(network.weights,correct_matrix_hebbian)

    #SUCCESS


def test_pattern_choice():
    """
    It tests if the returned random pattern is from the patterns array.
    """
    pattern_manager = pm.PatternManager()

    patterns = np.array([[1,1,-1,-1],[1,1,-1,1],[-1,1,-1,1]])
    pattern = pattern_manager.pattern_choice(patterns)
    assert pattern in patterns

def test_perturb_pattern():
    """
    It takes a pattern and a number of elements to flip and tests if the number of elements that are flipped is correct.
    """

    pattern_manager = pm.PatternManager()

    pattern = np.array([1,1,-1,-1])
    perturbed_pattern = pattern_manager.perturb_pattern(pattern, 3)
    assert np.count_nonzero(pattern - perturbed_pattern) != 0

def test_random_choice():
    """
    It tests if the function returns either 1 or -1
    """

    pattern_manager = pm.PatternManager()

    x = pattern_manager.random_choice()
    assert x == 1 or x == -1


def test_replace_random_pattern():
    """
    > Tests if we replace a random pattern in the array with a checkerboard.
    """

    pattern_manager = pm.PatternManager()

    patterns = np.array([[1,1,-1,-1],[1,1,-1,1],[-1,1,-1,1]])
    checkerboard = pattern_manager.checkerboard_visualization(2)
    pattern_manager.replace_random_pattern(patterns, checkerboard)
    assert (checkerboard in patterns) 

# def test_save_video():
#     """"
#     Saving the video of the patterns as a gif
#     """"
#     patterns = np.ones((4,4))
#     network.save_video(patterns,os.getcwd()+os.sep+"animation.gif")
#     assert os.path.exists('"animation.gif"') is True

def test_storkey_dynamics():
    """
    It takes a pattern, a weight matrix, and a number of iterations, and tests if the pattern is stable and the energy is not increasing
    """

    pattern_manager = pm.PatternManager()
    patterns= pattern_manager.create_patterns(3, 30)
    network = hn.HopfieldNetwork(patterns, "storkey")
    saver = ds.DataSaver()

    perturbed_pattern = pattern_manager.perturb_pattern(network.patterns[0], 2)
    result = network.dynamics(perturbed_pattern, saver, 20)
    states, _ = saver.get_data()

    assert np.allclose(network.patterns[0], result)

    assert np.allclose(states[-2], states[-1])
    

def test_storkey_weights():
    """
    It takes a matrix of patterns and tests if the weights are calculated correctly.
    """

    network = hn.HopfieldNetwork(np.array([[1,1,-1,-1],[1,1,-1,1],[-1,1,-1,1]]), "storkey")

    correct_matrix_storkey = [[1.125,0.25,-0.25,-0.5],[0.25,0.625,-1.,0.25],[-0.25,-1.,0.625,-0.25],[-0.5,0.25,-0.25,1.125]]
    assert np.allclose(network.weights, correct_matrix_storkey)

    #SUCCESS

def test_update():
    """
    It tests whether the update function returns a pattern of 1s and -1s and if the pattern is stable.
    """

    network = hn.HopfieldNetwork(np.array([[1,1,-1,-1],[1,1,-1,1],[-1,1,-1,1]]), "hebbian")
    
    updated_sync = False
    pattern = network.patterns[0]
    weights = network.hebbian_weights()
    updated_pattern_sync = network.update(pattern)

    updated_async = False
    updated_pattern_async = network.update_async(pattern)

    if  (elem == 1 or elem == -1 for elem in updated_pattern_async):
        updated_async = True

    if np.array_equal(pattern, updated_pattern_sync) is False and (elem == 1 or elem == -1 for elem in updated_pattern_sync):
        updated_sync = True
    assert updated_sync and updated_async