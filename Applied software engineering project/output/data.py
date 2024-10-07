import pandas as pd
import numpy as np
import capacity_plots


# Creating a dictionary with the key being the network size and the value being the number of
# patterns.

results_data = [{'network size': 10, 'weight rule': 'storkey', 'num patterns': 2, 'num perturb': 2, 'match frac': 0.8},
 {'network size': 10, 'weight rule': 'storkey', 'num patterns': 3, 'num perturb': 2, 'match frac': 1.0},
 {'network size': 10, 'weight rule': 'storkey', 'num patterns': 3, 'num perturb': 2, 'match frac': 0.8},
 {'network size': 10, 'weight rule': 'storkey', 'num patterns': 4, 'num perturb': 2, 'match frac': 0.8},
 {'network size': 10, 'weight rule': 'storkey', 'num patterns': 5, 'num perturb': 2, 'match frac': 0.6},
 {'network size': 10, 'weight rule': 'storkey', 'num patterns': 6, 'num perturb': 2, 'match frac': 0.2},
 {'network size': 10, 'weight rule': 'storkey', 'num patterns': 6, 'num perturb': 2, 'match frac': 0.7},
 {'network size': 10, 'weight rule': 'storkey', 'num patterns': 7, 'num perturb': 2, 'match frac': 0.2},
 {'network size': 10, 'weight rule': 'storkey', 'num patterns': 8, 'num perturb': 2, 'match frac': 0.3},
 {'network size': 10, 'weight rule': 'storkey', 'num patterns': 9, 'num perturb': 2, 'match frac': 0.3},
 {'network size': 18, 'weight rule': 'storkey', 'num patterns': 3, 'num perturb': 4, 'match frac': 0.9},
 {'network size': 18, 'weight rule': 'storkey', 'num patterns': 4, 'num perturb': 4, 'match frac': 0.9},
 {'network size': 18, 'weight rule': 'storkey', 'num patterns': 6, 'num perturb': 4, 'match frac': 0.7},
 {'network size': 18, 'weight rule': 'storkey', 'num patterns': 7, 'num perturb': 4, 'match frac': 0.5},
 {'network size': 18, 'weight rule': 'storkey', 'num patterns': 8, 'num perturb': 4, 'match frac': 0.4},
 {'network size': 18, 'weight rule': 'storkey', 'num patterns': 9, 'num perturb': 4, 'match frac': 0.6},
 {'network size': 18, 'weight rule': 'storkey', 'num patterns': 11, 'num perturb': 4, 'match frac': 0.1},
 {'network size': 18, 'weight rule': 'storkey', 'num patterns': 12, 'num perturb': 4, 'match frac': 0.4},
 {'network size': 18, 'weight rule': 'storkey', 'num patterns': 13, 'num perturb': 4, 'match frac': 0.1},
 {'network size': 18, 'weight rule': 'storkey', 'num patterns': 14, 'num perturb': 4, 'match frac': 0.1},
 {'network size': 34, 'weight rule': 'storkey', 'num patterns': 6, 'num perturb': 7, 'match frac': 0.9},
 {'network size': 34, 'weight rule': 'storkey', 'num patterns': 8, 'num perturb': 7, 'match frac': 0.7},
 {'network size': 34, 'weight rule': 'storkey', 'num patterns': 10, 'num perturb': 7, 'match frac': 0.7},
 {'network size': 34, 'weight rule': 'storkey', 'num patterns': 12, 'num perturb': 7, 'match frac': 0.8},
 {'network size': 34, 'weight rule': 'storkey', 'num patterns': 14, 'num perturb': 7, 'match frac': 0.5},
 {'network size': 34, 'weight rule': 'storkey', 'num patterns': 17, 'num perturb': 7, 'match frac': 0.1},
 {'network size': 34, 'weight rule': 'storkey', 'num patterns': 19, 'num perturb': 7, 'match frac': 0.2},
 {'network size': 34, 'weight rule': 'storkey', 'num patterns': 21, 'num perturb': 7, 'match frac': 0.2},
 {'network size': 34, 'weight rule': 'storkey', 'num patterns': 23, 'num perturb': 7, 'match frac': 0.1},
 {'network size': 34, 'weight rule': 'storkey', 'num patterns': 25, 'num perturb': 7, 'match frac': 0.0},
 {'network size': 63, 'weight rule': 'storkey', 'num patterns': 10, 'num perturb': 13, 'match frac': 1.0},
 {'network size': 63, 'weight rule': 'storkey', 'num patterns': 14, 'num perturb': 13, 'match frac': 0.8},
 {'network size': 63, 'weight rule': 'storkey', 'num patterns': 18, 'num perturb': 13, 'match frac': 0.9},
 {'network size': 63, 'weight rule': 'storkey', 'num patterns': 21, 'num perturb': 13, 'match frac': 0.7},
 {'network size': 63, 'weight rule': 'storkey', 'num patterns': 25, 'num perturb': 13, 'match frac': 0.2},
 {'network size': 63, 'weight rule': 'storkey', 'num patterns': 29, 'num perturb': 13, 'match frac': 0.1},
 {'network size': 63, 'weight rule': 'storkey', 'num patterns': 32, 'num perturb': 13, 'match frac': 0.1},
 {'network size': 63, 'weight rule': 'storkey', 'num patterns': 36, 'num perturb': 13, 'match frac': 0.1},
 {'network size': 63, 'weight rule': 'storkey', 'num patterns': 40, 'num perturb': 13, 'match frac': 0.0},
 {'network size': 63, 'weight rule': 'storkey', 'num patterns': 43, 'num perturb': 13, 'match frac': 0.0},
 {'network size': 116, 'weight rule': 'storkey', 'num patterns': 18, 'num perturb': 23, 'match frac': 1.0},
 {'network size': 116, 'weight rule': 'storkey', 'num patterns': 25, 'num perturb': 23, 'match frac': 1.0},
 {'network size': 116, 'weight rule': 'storkey', 'num patterns': 31, 'num perturb': 23, 'match frac': 0.9},
 {'network size': 116, 'weight rule': 'storkey', 'num patterns': 37, 'num perturb': 23, 'match frac': 0.5},
 {'network size': 116, 'weight rule': 'storkey', 'num patterns': 43, 'num perturb': 23, 'match frac': 0.5},
 {'network size': 116, 'weight rule': 'storkey', 'num patterns': 50, 'num perturb': 23, 'match frac': 0.4},
 {'network size': 116, 'weight rule': 'storkey', 'num patterns': 56, 'num perturb': 23, 'match frac': 0.0},
 {'network size': 116, 'weight rule': 'storkey', 'num patterns': 62, 'num perturb': 23, 'match frac': 0.0},
 {'network size': 116, 'weight rule': 'storkey', 'num patterns': 68, 'num perturb': 23, 'match frac': 0.0},
 {'network size': 116, 'weight rule': 'storkey', 'num patterns': 75, 'num perturb': 23, 'match frac': 0.0},
 {'network size': 215, 'weight rule': 'storkey', 'num patterns': 32, 'num perturb': 43, 'match frac': 1.0},
 {'network size': 215, 'weight rule': 'storkey', 'num patterns': 43, 'num perturb': 43, 'match frac': 1.0},
 {'network size': 215, 'weight rule': 'storkey', 'num patterns': 54, 'num perturb': 43, 'match frac': 1.0},
 {'network size': 215, 'weight rule': 'storkey', 'num patterns': 65, 'num perturb': 43, 'match frac': 0.6},
 {'network size': 215, 'weight rule': 'storkey', 'num patterns': 76, 'num perturb': 43, 'match frac': 0.3},
 {'network size': 215, 'weight rule': 'storkey', 'num patterns': 87, 'num perturb': 43, 'match frac': 0.2},
 {'network size': 215, 'weight rule': 'storkey', 'num patterns': 98, 'num perturb': 43, 'match frac': 0.1},
 {'network size': 215, 'weight rule': 'storkey', 'num patterns': 109, 'num perturb': 43, 'match frac': 0.0},
 {'network size': 215, 'weight rule': 'storkey', 'num patterns': 120, 'num perturb': 43, 'match frac': 0.0},
 {'network size': 215, 'weight rule': 'storkey', 'num patterns': 131, 'num perturb': 43, 'match frac': 0.0},
 {'network size': 397, 'weight rule': 'storkey', 'num patterns': 57, 'num perturb': 79, 'match frac': 1.0},
 {'network size': 397, 'weight rule': 'storkey', 'num patterns': 76, 'num perturb': 79, 'match frac': 1.0},
 {'network size': 397, 'weight rule': 'storkey', 'num patterns': 95, 'num perturb': 79, 'match frac': 1.0},
 {'network size': 397, 'weight rule': 'storkey', 'num patterns': 114, 'num perturb': 79, 'match frac': 1.0},
 {'network size': 397, 'weight rule': 'storkey', 'num patterns': 133, 'num perturb': 79, 'match frac': 0.5},
 {'network size': 397, 'weight rule': 'storkey', 'num patterns': 153, 'num perturb': 79, 'match frac': 0.1},
 {'network size': 397, 'weight rule': 'storkey', 'num patterns': 172, 'num perturb': 79, 'match frac': 0.0},
 {'network size': 397, 'weight rule': 'storkey', 'num patterns': 191, 'num perturb': 79, 'match frac': 0.0},
 {'network size': 397, 'weight rule': 'storkey', 'num patterns': 210, 'num perturb': 79, 'match frac': 0.0},
 {'network size': 397, 'weight rule': 'storkey', 'num patterns': 229, 'num perturb': 79, 'match frac': 0.0},
 {'network size': 733, 'weight rule': 'storkey', 'num patterns': 100, 'num perturb': 147, 'match frac': 1.0},
 {'network size': 733, 'weight rule': 'storkey', 'num patterns': 134, 'num perturb': 147, 'match frac': 1.0},
 {'network size': 733, 'weight rule': 'storkey', 'num patterns': 168, 'num perturb': 147, 'match frac': 1.0},
 {'network size': 733, 'weight rule': 'storkey', 'num patterns': 201, 'num perturb': 147, 'match frac': 1.0},
 {'network size': 733, 'weight rule': 'storkey', 'num patterns': 235, 'num perturb': 147, 'match frac': 0.7},
 {'network size': 733, 'weight rule': 'storkey', 'num patterns': 269, 'num perturb': 147, 'match frac': 0.3},
 {'network size': 733, 'weight rule': 'storkey', 'num patterns': 302, 'num perturb': 147, 'match frac': 0.0},
 {'network size': 733, 'weight rule': 'storkey', 'num patterns': 336, 'num perturb': 147, 'match frac': 0.1},
 {'network size': 733, 'weight rule': 'storkey', 'num patterns': 369, 'num perturb': 147, 'match frac': 0.0},
 {'network size': 733, 'weight rule': 'storkey', 'num patterns': 403, 'num perturb': 147, 'match frac': 0.0},
 {'network size': 1354, 'weight rule': 'storkey', 'num patterns': 178, 'num perturb': 271, 'match frac': 1.0},
 {'network size': 1354, 'weight rule': 'storkey', 'num patterns': 237, 'num perturb': 271, 'match frac': 1.0},
 {'network size': 1354, 'weight rule': 'storkey', 'num patterns': 297, 'num perturb': 271, 'match frac': 1.0},
 {'network size': 1354, 'weight rule': 'storkey', 'num patterns': 356, 'num perturb': 271, 'match frac': 0.8},
 {'network size': 1354, 'weight rule': 'storkey', 'num patterns': 415, 'num perturb': 271, 'match frac': 0.6},
 {'network size': 1354, 'weight rule': 'storkey', 'num patterns': 475, 'num perturb': 271, 'match frac': 0.6},
 {'network size': 1354, 'weight rule': 'storkey', 'num patterns': 534, 'num perturb': 271, 'match frac': 0.1},
 {'network size': 1354, 'weight rule': 'storkey', 'num patterns': 594, 'num perturb': 271, 'match frac': 0.0},
 {'network size': 1354, 'weight rule': 'storkey', 'num patterns': 653, 'num perturb': 271, 'match frac': 0.0},
 {'network size': 1354, 'weight rule': 'storkey', 'num patterns': 713, 'num perturb': 271, 'match frac': 0.0},
 {'network size': 2500, 'weight rule': 'storkey', 'num patterns': 315, 'num perturb': 500, 'match frac': 1.0},
 {'network size': 2500, 'weight rule': 'storkey', 'num patterns': 421, 'num perturb': 500, 'match frac': 1.0},
 {'network size': 2500, 'weight rule': 'storkey', 'num patterns': 526, 'num perturb': 500, 'match frac': 1.0},
 {'network size': 2500, 'weight rule': 'storkey', 'num patterns': 631, 'num perturb': 500, 'match frac': 1.0},
 {'network size': 2500, 'weight rule': 'storkey', 'num patterns': 737, 'num perturb': 500, 'match frac': 0.7},
 {'network size': 2500, 'weight rule': 'storkey', 'num patterns': 842, 'num perturb': 500, 'match frac': 0.3},
 {'network size': 2500, 'weight rule': 'storkey', 'num patterns': 947, 'num perturb': 500, 'match frac': 0.3},
 {'network size': 2500, 'weight rule': 'storkey', 'num patterns': 1053, 'num perturb': 500, 'match frac': 0.0},
 {'network size': 2500, 'weight rule': 'storkey', 'num patterns': 1158, 'num perturb': 500, 'match frac': 0.0},
 {'network size': 2500, 'weight rule': 'storkey', 'num patterns': 1263, 'num perturb': 500, 'match frac': 0.0}]


dict = {10: 3, 34: 6, 63: 18, 116: 31, 215: 54, 397: 114, 733: 201, 1354: 297, 2500: 631}

# Create a pandas DataFrame from results dictionary
df = pd.DataFrame(results_data)
# Save dataframe as an hdf5 file
#df.to_hdf(out_path, key='df')
# Pandas print the table in markdown format for easy pasting
print(df.to_markdown())
