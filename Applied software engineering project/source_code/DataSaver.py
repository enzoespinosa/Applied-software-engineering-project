from matplotlib.animation import Animation
import functions as hn
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.animation as Animation


class DataSaver:

    def __init__(self):
        """
        The function __init__() is a special function in Python classes. It is run as soon as an object
        of a class is instantiated. The method __init__() is perhaps the most important method in a class
        """
        self.states = np.array([])
        self.energies = np.array([])

    def reset(self):
        """
        It resets the states and energies arrays to empty arrays
        """
        self.states = np.array([])
        self.energies = np.array([])

    def store_iter(self, state, weights):
        """
        The function takes in a state and weights, and appends the state and the energy of the state to
        the states and energies arrays
        
        Parameters
        ----------
        state
            the current state of the system
        weights
            the weights of the neural network
        """
        if len(self.states) == 0:
            self.states = np.array([state])
        self.states = np.append(self.states,[state], axis=0)
        self.energies = np.append(self.energies, self.compute_energy(state, weights))

    def compute_energy(self, state, weights):
        """
        It computes the energy of a given state
        
        Parameters
        ----------
        state
            a vector of length N, where N is the number of nodes in the graph
        weights
            the weight matrix of the Ising model

        Returns
        -------
        The energy of the state.
        """
        return -0.5*np.dot(state, np.dot(weights, state.transpose()))

    def get_data(self):
        """
        It returns the states and energies of the system

        Returns
        -------
        The states and energies of the system.
        """
        return self.states, self.energies

    def save_video(self, out_path, img_shape):
        """
        The function takes in a list of images, and saves them as a video
        
        Parameters
        ----------
        out_path
            The path to save the video to
        img_shape
            The shape of the image to be saved
        """
        fig = plt.figure()
        plt.title("Convergence experiment")
        ims = []
        print("State SHAPE: ",self.states.shape)
        for state in self.states:
            print(state)
            state = state.reshape(img_shape)
            im = plt.imshow(state, cmap='gray', aspect='equal',origin= 'lower', extent=None, interpolation_stage=None, filternorm=True, 
            filterrad=4.0, resample=None, url=None, data=None)
            ims.append([im])
        ani = Animation.ArtistAnimation(fig, ims, interval=300, blit=True,repeat_delay=1000)
        ani.save(out_path, writer='pillow', fps=30)
        plt.show()
    
    def plot_energy(self, is_async):
        """
        It plots the energy evolution of the system
        
        Parameters
        ----------
        is_async
            True if asynchronous, False if synchronous
        """
        plt.figure()
        x_axis = np.arange(0,len(self.energies),1)
        y_axis = self.energies
        plt.title("Energy evolution")
        if is_async:
          plt.xlabel("Iterations in thousands")
        else:
          plt.xlabel("Iterations")
        plt.ylabel("Energy")
        plt.plot(x_axis, y_axis, color ="red")
        plt.show()