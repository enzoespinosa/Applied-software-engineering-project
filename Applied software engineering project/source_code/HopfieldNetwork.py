import numpy as np
import random


class HopfieldNetwork:

    def __init__(self, patterns, rule="hebbian"):
        """
        The function takes a list of  and a rule (either "hebbian" or "storkey") and returns a
        matrix of weights

        Parameters
        ----------

            a list of , each of which is a list of 1s and -1s
        rule
            the learning rule to use. Can be either "hebbian" or "storkey", defaults to hebbian
        """
        self.patterns = patterns
        if rule == "hebbian":
            self.weights = self.hebbian_weights()
        else :
            self.weights = self.storkey_weights()

    def hebbian_weights(self):
        """
        The function takes a list of  and returns the Hebbian weights matrix

        Parameters
        ----------

            a list of , each of which is a list of 1s and -1s

        Returns
        -------
        The Hebbian weights matrix.
        """
        if type(self.patterns) != np.ndarray:
            raise ValueError(
                "Given  are not a numpy array. Expected type: numpy.ndarray")

        n = len(self.patterns[0])
        m = len(self.patterns)
        weights = np.zeros((n, n))
        for el in self.patterns:
            weights += np.outer(el, el)
        w = weights/m
        np.fill_diagonal(w, 0)
        return w

    def storkey_weights(self):
        """
        The function takes in a numpy array of  and returns the weight matrix of the Hopfield
        network

        Parameters
        ----------

            a numpy array of , each of which is a list of 1s and -1s

        Returns
        -------
        The Storkey weights matrix.
        """
        N = np.size(self.patterns[0])
        W, H = np.zeros((N, N)), np.zeros((N, N))
        previous_storkey_weights = np.zeros((N, N))
        for mu in range(np.shape(self.patterns)[0]):
            matrix = np.tile(self.patterns[mu], (N, 1)).T
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
 

    def update(self, state):
        """
        The function takes a state vector and returns the synchronously updated state vector

        Parameters
        ----------
        state
            the state vector

        Returns
        -------
        The updated state vector
        """
        next_pattern = np.array(
            [1 if elem >= 0 else -1 for elem in np.dot(self.weights, state)])
        return next_pattern

    def update_async(self, state):
        """
        The function takes a state vector and returns the asynchronously updated state vector

        Parameters
        ----------
        state
            the state vector
        max_iter
            the maximum number of iterations to run the update for

        Returns
        -------
        The updated state vector
        """
        idx = random.randint(0, len(state)-1)
        new_state = state.copy()
        new_state[idx] = 1 if np.dot(self.weights, state)[idx] >= 0 else -1
        return new_state

    def dynamics_async(self, state, saver, max_iter=1000, convergence_num_iter=100, skip=10):
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
            next_state = self.update_async(prev_state)
            counter = counter + \
                1 if np.array_equal(next_state, prev_state) else 0
            if counter == convergence_num_iter:
                print("Convergence has been achieved in :", i+1, " steps ")
                saver.store_iter(next_state, self.weights)
                dynamic_async_history = np.append(
                    dynamic_async_history, [next_state], axis=0)
                break
            if i % skip == 0:
                print("Iteration: ", i)
                saver.store_iter(next_state, self.weights)
                dynamic_async_history = np.append(
                    dynamic_async_history, [next_state], axis=0)

            prev_state = next_state
        return dynamic_async_history

    def dynamics(self, state, saver, max_iter=20):
        """
        The function takes a state vector and returns the synchronoucly updated state vector

        Parameters
        ----------
        state
            the state vector
        saver
            the saver object
        max_iter
            the maximum number of iterations to run the update for

        Returns
        -------
        The synchronously updated state vector
        """
        prev_state = state.copy()
        for i in range(1,max_iter+1):
            next_state = self.update(prev_state)
            if np.array_equal(next_state, prev_state):
                print("Convergence achieved in {} iterations".format(i))
                #print("Final state: {}".format(next_state))
                saver.store_iter(next_state, self.weights)
                return next_state
            saver.store_iter(next_state, self.weights)
            prev_state = next_state
        print("No convergence")
        return next_state

        new_state = state.copy()
        old_state = np.zeros_like(state)
        saver.store_iter(new_state, self.weights)
        convergence = 0
        counter = 0

        while (counter < max_iter) and convergence < 2 and (not np.array_equal(new_state, old_state)):
            old_state = new_state
            new_state = self.update(old_state)
            counter += 1
            saver.store_iter(new_state, self.weights)
            if np.array_equal(new_state, old_state):
                convergence += 1
            else:
                convergence = 0
            if convergence == 0:
                print ("Convergence not reached.")
            else:
                print ("Convergence reached in " + str(convergence) + " iterations. ")
        saver.store_iter(new_state, self.weights)
        return convergence     