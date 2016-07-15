from COINRandomZero import COINRandomZero
import sys
sys.path.append("..")
import Report

graph_file = "subnethept.txt"
experiment_name = "noncontextual_" + graph_file[:-4]
rounds = 1250
seed_size = 50

coinrandomzero = COINRandomZero(seed_size, graph_file, rounds)
coinrandomzero()

Report.report("coinrandomzero", coinrandomzero, experiment_name)
