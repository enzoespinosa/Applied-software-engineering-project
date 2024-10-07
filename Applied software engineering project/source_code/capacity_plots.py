import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import DataSaver as ds
import PatternManager as pm
import HopfieldNetwork as hn
import Experiment

def plot_rates(results,sizes, weighs_rule):
    """
    It plots the match fraction as a function of the number of patterns for a given network size and
    weight rule
    
    Parameters
    ----------
    results
        a list of dictionaries, each dictionary contains the results of a single simulation
    sizes
        the network sizes to plot
    weighs_rule
        "random" or "hebbian"
    """
    i=0
    for s in sizes:

        match_frac = []
        t = []
        for result in results:
            if result["network size"] == s and result["weight rule"] == weighs_rule:
                match_frac.append(result["match frac"])
                t.append(result["num patterns"])
        plt.plot(t, match_frac)
        plt.title("N = "+str(s))       
        plt.xlabel("Number of patterns")
        plt.ylabel("Match fraction")
        plt.ylim(0,1)
        plt.xlim(t[0],t[-1])
        plt.show()
        i+=1
    



    

def plot_capacities(experimental, sizes, weighs_rule):
    """
    It plots the theoretical and experimental capacities for a given rule

    Parameters
    ----------
    experimental
        a list of the experimental capacities for each network size
    sizes
        the sizes of the networks to test
    weighs_rule
        "hebbian" or "storkey" -> the learning rules
    """

    fig,ax = plt.subplots()
    fig.suptitle("Capacity for " + str(weighs_rule))
    ssss = np.linspace(sizes[0], sizes[-1],100)
    if weighs_rule == "hebbian":
        theoretical = hebbian_capacity(ssss)
    else:
        theoretical = storkey_capacity(ssss)

    ax.plot(ssss, theoretical, label="Theoretical")
    ax.plot(sizes, experimental, label="Experimental")
    ax.set_xlabel("Network size")
    ax.set_ylabel("Capacity")

    ax.set_xlim(sizes[0],sizes[-1])
    plt.show()

def storkey_capacity(s):
    """
    It computes the capacity of a Storkey network with s hidden units
    
    Parameters
    ----------
    s
        the number of units in the layer

    Returns
    -------
    The capacity of the network.
    """
    return s / np.sqrt(2 * np.log(s))

def hebbian_capacity(s):
    """
    The capacity of a Hebbian network is the number of patterns it can store, divided by the number of
    neurons in the network
    
    Parameters
    ----------
    s
        number of synapses per neuron

    Returns
    -------
    The capacity of the network.
    """
    return s / (2 * np.log(s))    
        

def plot_perturbation(results,parturbations,weighs_rule, size):
    """
    It plots the results of the perturbation analysis
    
    Parameters
    ----------
    results
        the list of results for each perturbation
    parturbations
        the percentage of perturbations to be applied to the network
    weighs_rule
        the rule used to generate the weights matrix
    size
        the number of neurons in the network
    """

    plt.plot(parturbations, results)
    plt.title("N = "+str(size)+" ("+str(weighs_rule)+")")
    plt.xlabel("Percentage of perturbations [%]")
    plt.ylabel("Retrieved patterns as a function of perturbation percentage " + str(n) + " neurons using " + str(weighs_rule) + "rule")
    plt.ylim(0,1)
    #plt.xlim(0,100)
    plt.show()

   