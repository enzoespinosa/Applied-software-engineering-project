import os
import time
from timeit import default_timer as timer
import data
import numpy as np


import functions
# from dynamics_cython import dynamics
# from dynamics_async_cython import dynamics_async

import DataSaver
import HopfieldNetwork
import PatternManager

import Experiment
import capacity_plots

def main():
    # print("Generating patterns")
    # patterns = HopfieldNetwork.generate_patterns(50, 2500)
    # # print(patterns)

    # print("Generating weights")
    # start_time = time.time()
    # hebbian_weights = HopfieldNetwork.hebbian_weights(patterns)
    # print("time spent computing the hebbian weights: ", time.time()-start_time)

    # start_time = time.time()
    # storkey_weights = HopfieldNetwork.storkey_weights(patterns)
    # print("time spent computing the storkey weights: ", time.time()-start_time)

    # chosen_pattern = HopfieldNetwork.pattern_choice(patterns)
    # perturbed_pattern = HopfieldNetwork.perturb_pattern(chosen_pattern, 1000)

    # start_time = time.time()
    # hist_hebbian_synch = dynamics(
    #     perturbed_pattern, hebbian_weights, 20)
    # print("time spent computing the hebbian dynamics synchronously: ",
    #       time.time()-start_time)
    # print("Is it correct? ", np.array_equal(
    #     hist_hebbian_synch[-1], chosen_pattern))
    # energy_history_hebbian_synch = HopfieldNetwork.energy_evolution(
    #     hist_hebbian_synch, hebbian_weights)
    # if HopfieldNetwork.energy_is_not_increasing(energy_history_hebbian_synch):
    #     print("The energy is not increasing - hebbian synch")
    # else:
    #     print("ERROR: The energy is increasing - hebbian synch")

    # print(energy_history_hebbian_synch)
    # HopfieldNetwork.energy_plot(energy_history_hebbian_synch)

    # start_time = time.time()
    # hist_storkey_synch = dynamics(
    #     perturbed_pattern, storkey_weights, 20)
    # print("time spent computing the storkey dynamics synchronously: ",
    #       time.time()-start_time)
    # print("Is it correct? ", np.array_equal(
    #     hist_storkey_synch[-1], chosen_pattern))
    # energy_history_storkey_synch = HopfieldNetwork.energy_evolution(
    #     hist_storkey_synch, storkey_weights)
    # if HopfieldNetwork.energy_is_not_increasing(energy_history_storkey_synch):
    #     print("The energy is not increasing - storkey synch")
    # else:
    #     print("ERROR: The energy is increasing - storkey synch")

    # start_time = time.time()
    # hist_hebbian_asynch = dynamics_async(
    #     perturbed_pattern, hebbian_weights, 30000)
    # print("time spent computing the hebbian dynamics asynchronously: ",
    #       time.time()-start_time)
    # print("Is it correct? ", np.array_equal(
    #     hist_hebbian_asynch[-1], chosen_pattern))
    # print(hist_hebbian_asynch.shape)
    # energy_history_hebbian_asynch = HopfieldNetwork.energy_evolution(
    #     hist_hebbian_asynch, hebbian_weights)
    # if HopfieldNetwork.energy_is_not_increasing(energy_history_hebbian_asynch):
    #     print("The energy is not increasing - hebbian asynch")
    # else:
    #     print("ERROR: The energy is increasing - hebbian asynch")

    # HopfieldNetwork.energy_plot(energy_history_hebbian_asynch, is_async=True)

    # start_time = time.time()
    # hist_storkey_asynch = dynamics_async(
    #     perturbed_pattern, storkey_weights, 30000)
    # print("time spent computing the storkey dynamics asynchronously: ",
    #       time.time()-start_time)
    # print("Is it correct? ", np.array_equal(
    #     hist_storkey_asynch[-1], chosen_pattern))
    # energy_history_storkey_asynch = HopfieldNetwork.energy_evolution(
    #     hist_storkey_asynch, storkey_weights)
    # if HopfieldNetwork.energy_is_not_increasing(energy_history_storkey_asynch):
    #     print("The energy is not increasing - storkey asynch")
    # else:
    #     print("ERROR: The energy is increasing - storkey asynch")



    # checkerboard_pattern = HopfieldNetwork.checkerboard_visualization(50)
    # patterns = HopfieldNetwork.replace_random_pattern(
    #     patterns, checkerboard_pattern)
    # weights = HopfieldNetwork.hebbian_weights(patterns)

    # perturbed_checkerboard_pattern = HopfieldNetwork.perturb_pattern(
    #     checkerboard_pattern, 1000)
    # hist_async = dynamics_async(
    #     perturbed_checkerboard_pattern, weights, 30000)
    # hist_sync = dynamics(
    #     perturbed_checkerboard_pattern, weights)

    # reshaped_asynch_states = np.reshape(hist_async, (-1, 50, 50))
    # reshaped_synch_states = np.reshape(hist_sync, (-1, 50, 50))
    # HopfieldNetwork.save_video(
    #     reshaped_asynch_states, os.getcwd()+os.sep+"animation_asynch.gif")
    # HopfieldNetwork.save_video(
    #     reshaped_synch_states, os.getcwd()+os.sep+"animation_synch.gif")

    # # V7 start
    saver = DataSaver.DataSaver()
    print("Creating patterns...")
    pattern_manager = PatternManager.PatternManager()
    patterns = pattern_manager.create_patterns(50, 2500)
    network = HopfieldNetwork.HopfieldNetwork(patterns, "hebbian")
    chosenPattern = pattern_manager.pattern_choice(patterns)
    perturbed_pattern = pattern_manager.perturb_pattern(chosenPattern, 500)
    # Synchronous dynamics
    #hist_sync = network.dynamics(perturbed_pattern, saver, 20)
    #saver.plot_energy(is_async=False)
    #saver.reset()
    # Asynchronous dynamics
   # hist_async = network.dynamics_async(perturbed_pattern, saver, 1000, 100, 10)
    #saver.plot_energy(is_async=True)
    #saver.reset()
    # Checkerboard simulation
    print ("Checkboard")
    checkerboard_pattern = pattern_manager.checkerboard_visualization(50)
    patterns = pattern_manager.replace_random_pattern(patterns, checkerboard_pattern)
    checkerboard_network = HopfieldNetwork.HopfieldNetwork(patterns, "hebbian")
    # Synchronous dynamics
    print ("Sychronous")
    perturbed_checkerboard = pattern_manager.perturb_pattern(checkerboard_pattern, 3000)
    hist_sync = checkerboard_network.dynamics(perturbed_checkerboard, saver, 20)
    saver.plot_energy(is_async=False)
    saver.save_video(os.getcwd()+os.sep+"output"+os.sep+"animation_synch.gif",(50,50))
    print("C'était animation sync")
    saver.reset()
    # Asynchronous dynamics
    print ("Asynchronous")
    hist_async = checkerboard_network.dynamics_async(perturbed_checkerboard, saver, 100000, 10000, 1000)
    saver.plot_energy(is_async=False)
    saver.save_video(os.getcwd()+os.sep+"output"+os.sep+"animation_asynch.gif",(50,50))
    print("C'était animation async")
    saver.plot_energy(is_async=True)
    saver.reset()
    # #  # V7 end



    
    # sizes=[10, 18, 34, 63, 116, 215, 397, 733, 1354, 2500]
    # # networks = np.array([])
    # # results = np.array([])
    # # cs=[]
    #i=10
    # # for size in sizes:     
    # #     C = size / ( 2 * np.log(size))
    # #     cs.append(C)
    # #     possible_t = np.linspace(0.5 * C, 2 * C, 10).astype(int)
    # #     print(possible_t)
    # #     for t in possible_t:
    # #         results_dic_hebbian = Experiment.experiment(size, t, "hebbian" , int(0.2 * size),10*i,100)
    # #         results = np.append(results, results_dic_hebbian)
    # #     i-=1    

    # tuples =[(i['network size'],i['num patterns'],i['match frac']) for i in results]
    # maxs = {}
    # for i in tuples:
    #     if i[2] >= 0.9:
    #         if i[0] not in maxs:
    #             maxs[i[0]] = i[1]
    #         else:
    #             if i[1] > maxs[i[0]]:
    #                 maxs[i[0]] = i[1]

    # print("Theorie:",cs)
    # print("Pratique:",maxs)
    # #maxs should be an array sorted by size

    # maxs = [maxs[i] for i in sizes]
    # print(results)
    # print(cs)

    # capacity_plots.plot_rates(results,sizes,"hebbian")
    # capacity_plots.plot_capacities(maxs,sizes,"hebbian")
        
    # results = np.array([])
    # cs=[]
    # for size in sizes:
    #     C = size / np.sqrt(2*np.log(size))
    #     cs.append(C)
    #     possible_t = np.linspace(0.5 * C, 2 * C, 10).astype(int)
    #     for t in possible_t:
    #         #print("t:",t)
    #         results_dic_storkey =Experiment.experiment(size, t, "storkey", np.rint(0.2 * size).astype(int),10,100)
    #         print("result",results_dic_storkey)
    #         results = np.append(results, results_dic_storkey)
    # print(results)
    # print(cs)

    # results=data.results_data
    # tuples =[(i['network size'],i['num patterns'],i['match frac']) for i in results]
    # maxs = {}
    # print("Tuples:",[i for i in tuples if i[0]==18])
    # for i in tuples:
    #     if i[2] >= 0.9:
    #         if i[0] not in maxs:
    #             print("network size",i[0])
    #             maxs[i[0]] = i[1]
    #         else:
    #             if i[1] > maxs[i[0]]:
    #                 maxs[i[0]] = i[1]

    # #print("Theorie:",cs)
    # print("Pratique:",maxs)
    # #maxs should be an array sorted by size
    # print("maxs ",maxs)
    # maxs = [maxs[i] for i in sizes]
    
    # print(results)
    # #print(cs)
    # capacity_plots.plot_rates(results,sizes,"storkey")
    # capacity_plots.plot_capacities(maxs,sizes,"storkey")
        

    # sizes=[10, 18, 34, 63, 116, 215, 397, 733, 1354, 2500]
    
    # for size in sizes:
    #     results = np.array([])
    #     #for each size,compute the success rate for t=2 but with different perturbation %
    #     parturbations = np.linspace(0, 1, 21)
    #     for i in parturbations:
    #         print("i*size",(i*size).astype(int))
    #         results_dic_hebbian = Experiment.experiment(size, 2, "storkey" , (i*size).astype(int),10,100)
    #         results = np.append(results, results_dic_hebbian)
    #     print("results",results)
    #     #then add the results to an array and plot the results
    #     array=[ i['match frac'] for i in results]
    #     capacity_plots.plot_perturbation(array,parturbations,"storkey",size)

    

    
        


if __name__ == "__main__":
    start = timer()
    main()
    end = timer()
    duration = end - start
    print("code running duration ", duration)
