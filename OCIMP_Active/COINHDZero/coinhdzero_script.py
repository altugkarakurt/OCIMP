from COINHDZero import COINHDZero
import sys
sys.path.append("..")
import Report

graph_file = "subnethept.txt"
experiment_name = "noncontextual_" + graph_file[:-4]
rounds = 1250
seed_size = 50
cost = 0.1

coinhdzero = COINHDZero(seed_size, graph_file, rounds, cost)
coinhdzero()

Report.report("coinhdzero", coinhdzero, experiment_name)
