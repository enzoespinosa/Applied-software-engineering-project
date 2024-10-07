import cProfile
import pstats
from main import main

# Profiling the code and printing the stats.

cProfile.run("main()", "restats")
p = pstats.Stats('restats')
p.sort_stats('cumulative').print_stats(20)
# We can change the number if we want to visualize more stats