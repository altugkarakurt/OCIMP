from OIMZero import OIMZero
import sys
sys.path.append("..")
import Report

graph_file = "nethept.txt"
experiment_name = "noncontextual_" + graph_file[:-4]
rounds = 2500
seed_size = 50

oimzero = OIMZero(seed_size, graph_file, rounds)
oimzero()

Report.report("oimzero", oimzero, experiment_name)
