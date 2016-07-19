from COINZero import COINZero
import sys
sys.path.append("..")
import Report

graph_file = "subnethept.txt"
experiment_name = "noncontextual_" + graph_file[:-4]
rounds = 1250
seed_size = 50
cost = 0.1

coinzero = COINZero(seed_size, graph_file, rounds, cost)
coinzero()

Report.report("coinzero", coinzero, experiment_name)
