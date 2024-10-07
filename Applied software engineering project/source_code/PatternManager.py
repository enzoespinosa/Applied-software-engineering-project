import itertools
import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np



class PatternManager:

    def random_choice(self):
        """
        It creates a random choice between -1 and 1

        Returns
        -------
        A random choice between -1 and 1
        """
        return np.random.choice([-1, 1])

    def create_pattern(self, N):
        """
        It creates a random vector of length N

        Parameters
        ----------
        N
            The length of the vector

        Returns
        -------
        A random vector of length N
        """

        pattern = np.ones(N)
        for i in range(0, N-1):
            pattern[i] = self.random_choice()
        return pattern

    def create_patterns(self, M, N):
        """
        It creates M patterns of length N

        Parameters
        ----------
        N
            The length of the patterns
        M
            The number of patterns

        Returns
        -------
        A matrix of M patterns of length N
        """
        patterns = np.zeros(0)
        for n in range(M):
            patterns = np.append(patterns, self.create_pattern(N))
        return np.reshape(patterns, (M, N))

    def perturb_pattern(self, pattern, k):
        """
        It perturbs a pattern by flipping k elements

        Parameters
        ----------
        pattern
            The pattern to be perturbed
        k
            The number of elements to be flipped
        
        Returns
        -------
        The perturbed pattern
        """
        new_pattern = pattern.copy()
        pattern_max_idx = len(pattern)-1
        for _ in itertools.repeat(None, k):
            idx = random.randint(0, pattern_max_idx)
            new_pattern[idx] = -new_pattern[idx]
        return new_pattern

    def pattern_match(self, memorized_patterns, pattern):
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

    def checkerboard_visualization(self,size):
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

    def pattern_choice(self, patterns):
        """
        It chooses a random pattern from the list of memorized patterns

        Parameters
        ----------
        memorized_patterns
            a list of patterns that have been memorized

        Returns
        -------
        A random pattern from the list of memorized patterns
        """
        return patterns[random.randint(0,len(patterns)-1)]

    def replace_random_pattern(self, patterns, pattern):
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
        patterns[idx] = pattern
        return patterns
