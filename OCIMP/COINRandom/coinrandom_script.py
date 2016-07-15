from COINRandom import COINRandom
import sys
sys.path.append("..")
import Report

graph_file = "nethept.txt"
experiment_name = "noncontextual_" + graph_file[:-4]
rounds = 1250
seed_size = 50

coinrandom = COINRandom(seed_size, graph_file, rounds)
coinrandom()

Report.report("coinrandom", coinrandom, experiment_name)
