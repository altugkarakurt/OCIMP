from COINRandomRandom import COINRandomRandom
import sys
sys.path.append("..")
import Report

graph_file = "nethept.txt"
experiment_name = "noncontextual_" + graph_file[:-4]
rounds = 1250
seed_size = 50

coinrandomrandom = COINRandomRandom(seed_size, graph_file, rounds)
coinrandomrandom()

Report.report("coinrandomrandom", coinrandomrandom, experiment_name)
