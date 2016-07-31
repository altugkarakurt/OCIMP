from OIMRandom import OIMRandom
import sys
sys.path.append("..")
import Report

graph_file = "subnethept.txt"
contextual = True
experiment_name = "contextual_" if(iscontextual) else "noncontextual_"
experiment_name += graph_file[:-4]
rounds = 1250
seed_size = 50

oimrandom = OIMRandom(seed_size, graph_file, rounds, iscontextual)
oimrandom()

Report.report("oimrandom", oimrandom, experiment_name)
