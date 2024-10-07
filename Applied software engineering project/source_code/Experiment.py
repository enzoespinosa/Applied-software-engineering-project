from __future__ import division
import numpy as np
import HopfieldNetwork as hn
import PatternManager as pm
import DataSaver as ds
import functions as fn

def experiment(size, num_patterns, weight_rule, num_perturb, num_trials=10, max_iter=100):
    """
    It takes in a bunch of parameters, creates a network, and then runs a bunch of trials to see how
    many times the network can retrieve a pattern
    
    Parameters
    ----------
    size
        the size of the network
    num_patterns
        the number of patterns to be stored in the network
    weight_rule 
        "hebbian" or "storkey"
    num_perturb
        the number of bits to flip in the chosen pattern
    num_trials
        number of times to run the experiment, defaults to 10 (optional)
    max_iter
        the maximum number of iterations to run the network for, defaults to 100 (optional)

    Returns
    -------
    A dictionary with the network size, weight rule, number of patterns, number of
    perturbations, and the match fraction.
    """
    counter = 0

    saver = ds.DataSaver()
    pattern_manager = pm.PatternManager()
    patterns = pattern_manager.create_patterns(num_patterns, size)
    network =hn.HopfieldNetwork(patterns, weight_rule)
    #print("Weights: ",network.weights)
    for n in range(num_trials) :
        print(str(num_patterns)+"------"+str(n)+"--------")
        chosen_pattern = pattern_manager.pattern_choice(patterns)
        perturbed_pattern = pattern_manager.perturb_pattern(chosen_pattern, num_perturb)
        retrieved_state = network.dynamics_sync(perturbed_pattern, saver, max_iter)
        if np.array_equal(retrieved_state, chosen_pattern) :
            print("Match found")
            counter += 1

    match_frac =  counter / num_trials
    results_dict = {"network size": size, "weight rule": weight_rule, "num patterns": num_patterns, "num perturb": num_perturb, "match frac": match_frac}
    return results_dict