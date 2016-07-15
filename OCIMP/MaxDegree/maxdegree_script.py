from MaxDegree import MaxDegree
import sys
sys.path.append("..")
import Report

graph_file = "nethept.txt"
experiment_name = "noncontextual_" + graph_file[:-4]
rounds = 2500
seed_size = 50

maxdegree = MaxDegree(seed_size, graph_file, rounds)
maxdegree()

Report.report("maxdegree", maxdegree, experiment_name)
