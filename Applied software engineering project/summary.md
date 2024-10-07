# PROJECT SUMMARY - BIO-210 Team 31: Hopfield Network



# Explanation and recap
The goal of our project is to simulate a neural network, according to the assumption that "neurons that fire together, wire together".
We implemented two different learning rules, essentially matrices that shaped the outcomes of our network using updating functions.
We used the Hebbian and Storkey models to simulate the behavior of lists of neurons, and to predict the behavior of the system when it was perturbed : we were interested into how it converged, if it was capable of retrieving the initial state and how long it took to do so.
Furthemore, we were also interested in the capacity of the network - or, how the network was capable to store many difeent patterns, depending on the network size, the weight rule used, the number of patterns and the number of perturbations per pattern. A meaningful output is the match fraction, which reresents the ration of th number of righltly retrieved patterns over the total of patterns stored in the network.



# Markdown table for results dictionnary
This is our results dictionnaty, in markup version using pandas library (see in data.py):

|    |   network size | weight rule   |   num patterns |   num perturb |   match frac |
|---:|---------------:|:--------------|---------------:|--------------:|-------------:|
|  0 |             10 | storkey       |              2 |             2 |          0.8 |
|  1 |             10 | storkey       |              3 |             2 |          1   |
|  2 |             10 | storkey       |              3 |             2 |          0.8 |
|  3 |             10 | storkey       |              4 |             2 |          0.8 |
|  4 |             10 | storkey       |              5 |             2 |          0.6 |
|  5 |             10 | storkey       |              6 |             2 |          0.2 |
|  6 |             10 | storkey       |              6 |             2 |          0.7 |
|  7 |             10 | storkey       |              7 |             2 |          0.2 |
|  8 |             10 | storkey       |              8 |             2 |          0.3 |
|  9 |             10 | storkey       |              9 |             2 |          0.3 |
| 10 |             18 | storkey       |              3 |             4 |          0.9 |
| 11 |             18 | storkey       |              4 |             4 |          0.9 |
| 12 |             18 | storkey       |              6 |             4 |          0.7 |
| 13 |             18 | storkey       |              7 |             4 |          0.5 |
| 14 |             18 | storkey       |              8 |             4 |          0.4 |
| 15 |             18 | storkey       |              9 |             4 |          0.6 |
| 16 |             18 | storkey       |             11 |             4 |          0.1 |
| 17 |             18 | storkey       |             12 |             4 |          0.4 |
| 18 |             18 | storkey       |             13 |             4 |          0.1 |
| 19 |             18 | storkey       |             14 |             4 |          0.1 |
| 20 |             34 | storkey       |              6 |             7 |          0.9 |
| 21 |             34 | storkey       |              8 |             7 |          0.7 |
| 22 |             34 | storkey       |             10 |             7 |          0.7 |
| 23 |             34 | storkey       |             12 |             7 |          0.8 |
| 24 |             34 | storkey       |             14 |             7 |          0.5 |
| 25 |             34 | storkey       |             17 |             7 |          0.1 |
| 26 |             34 | storkey       |             19 |             7 |          0.2 |
| 27 |             34 | storkey       |             21 |             7 |          0.2 |
| 28 |             34 | storkey       |             23 |             7 |          0.1 |
| 29 |             34 | storkey       |             25 |             7 |          0   |
| 30 |             63 | storkey       |             10 |            13 |          1   |
| 31 |             63 | storkey       |             14 |            13 |          0.8 |
| 32 |             63 | storkey       |             18 |            13 |          0.9 |
| 33 |             63 | storkey       |             21 |            13 |          0.7 |
| 34 |             63 | storkey       |             25 |            13 |          0.2 |
| 35 |             63 | storkey       |             29 |            13 |          0.1 |
| 36 |             63 | storkey       |             32 |            13 |          0.1 |
| 37 |             63 | storkey       |             36 |            13 |          0.1 |
| 38 |             63 | storkey       |             40 |            13 |          0   |
| 39 |             63 | storkey       |             43 |            13 |          0   |
| 40 |            116 | storkey       |             18 |            23 |          1   |
| 41 |            116 | storkey       |             25 |            23 |          1   |
| 42 |            116 | storkey       |             31 |            23 |          0.9 |
| 43 |            116 | storkey       |             37 |            23 |          0.5 |
| 44 |            116 | storkey       |             43 |            23 |          0.5 |
| 45 |            116 | storkey       |             50 |            23 |          0.4 |
| 46 |            116 | storkey       |             56 |            23 |          0   |
| 47 |            116 | storkey       |             62 |            23 |          0   |
| 48 |            116 | storkey       |             68 |            23 |          0   |
| 49 |            116 | storkey       |             75 |            23 |          0   |
| 50 |            215 | storkey       |             32 |            43 |          1   |
| 51 |            215 | storkey       |             43 |            43 |          1   |
| 52 |            215 | storkey       |             54 |            43 |          1   |
| 53 |            215 | storkey       |             65 |            43 |          0.6 |
| 54 |            215 | storkey       |             76 |            43 |          0.3 |
| 55 |            215 | storkey       |             87 |            43 |          0.2 |
| 56 |            215 | storkey       |             98 |            43 |          0.1 |
| 57 |            215 | storkey       |            109 |            43 |          0   |
| 58 |            215 | storkey       |            120 |            43 |          0   |
| 59 |            215 | storkey       |            131 |            43 |          0   |
| 60 |            397 | storkey       |             57 |            79 |          1   |
| 61 |            397 | storkey       |             76 |            79 |          1   |
| 62 |            397 | storkey       |             95 |            79 |          1   |
| 63 |            397 | storkey       |            114 |            79 |          1   |
| 64 |            397 | storkey       |            133 |            79 |          0.5 |
| 65 |            397 | storkey       |            153 |            79 |          0.1 |
| 66 |            397 | storkey       |            172 |            79 |          0   |
| 67 |            397 | storkey       |            191 |            79 |          0   |
| 68 |            397 | storkey       |            210 |            79 |          0   |
| 69 |            397 | storkey       |            229 |            79 |          0   |
| 70 |            733 | storkey       |            100 |           147 |          1   |
| 71 |            733 | storkey       |            134 |           147 |          1   |
| 72 |            733 | storkey       |            168 |           147 |          1   |
| 73 |            733 | storkey       |            201 |           147 |          1   |
| 74 |            733 | storkey       |            235 |           147 |          0.7 |
| 75 |            733 | storkey       |            269 |           147 |          0.3 |
| 76 |            733 | storkey       |            302 |           147 |          0   |
| 77 |            733 | storkey       |            336 |           147 |          0.1 |
| 78 |            733 | storkey       |            369 |           147 |          0   |
| 79 |            733 | storkey       |            403 |           147 |          0   |
| 80 |           1354 | storkey       |            178 |           271 |          1   |
| 81 |           1354 | storkey       |            237 |           271 |          1   |
| 82 |           1354 | storkey       |            297 |           271 |          1   |
| 83 |           1354 | storkey       |            356 |           271 |          0.8 |
| 84 |           1354 | storkey       |            415 |           271 |          0.6 |
| 85 |           1354 | storkey       |            475 |           271 |          0.6 |
| 86 |           1354 | storkey       |            534 |           271 |          0.1 |
| 87 |           1354 | storkey       |            594 |           271 |          0   |
| 88 |           1354 | storkey       |            653 |           271 |          0   |
| 89 |           1354 | storkey       |            713 |           271 |          0   |
| 90 |           2500 | storkey       |            315 |           500 |          1   |
| 91 |           2500 | storkey       |            421 |           500 |          1   |
| 92 |           2500 | storkey       |            526 |           500 |          1   |
| 93 |           2500 | storkey       |            631 |           500 |          1   |
| 94 |           2500 | storkey       |            737 |           500 |          0.7 |
| 95 |           2500 | storkey       |            842 |           500 |          0.3 |
| 96 |           2500 | storkey       |            947 |           500 |          0.3 |
| 97 |           2500 | storkey       |           1053 |           500 |          0   |
| 98 |           2500 | storkey       |           1158 |           500 |          0   |
| 99 |           2500 | storkey       |           1263 |           500 |          0   |



# Visualization
For the simple checkboard template, see: [chekboard](output/aaaa.gif)
For the asynchronous update way, see:[asynchronous animation](output/animation_asynch.gif)
For the synchronous update way, see: [synchronous animation](output/animation_synch.gif)



# Additional
For more informations, please take a look at our [README](README.md)!
Finally, thank you very much to the TAs for the help given all along, and to professor Mathis!