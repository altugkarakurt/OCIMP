from OIMRandom import OIMRandom
import sys
sys.path.append("..")
import Report

graph_file = "nethept.txt"
experiment_name = "noncontextual_" + graph_file[:-4]
rounds = 1250
seed_size = 50

oimrandom = OIMRandom(seed_size, graph_file, rounds)
oimrandom()

Report.report("oimrandom", oimrandom, experiment_name)
