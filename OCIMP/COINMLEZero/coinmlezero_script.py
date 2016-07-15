from COINMLEZero import COINMLEZero
import sys
sys.path.append("..")
import Report

graph_file = "nethept.txt"
experiment_name = "noncontextual_" + graph_file[:-4]
rounds = 1250
seed_size = 50

coinmlezero = COINMLEZero(seed_size, graph_file, rounds)
coinmlezero()

Report.report("coinmlezero", coinmlezero, experiment_name)
