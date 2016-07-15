from ThompsonIMZero import ThompsonIMZero
import sys
sys.path.append("..")
import Report

graph_file = "nethept.txt"
experiment_name = "noncontextual_" + graph_file[:-4]
rounds = 2500
seed_size = 50

thompsonzero = ThompsonIMZero(seed_size, graph_file, rounds)
thompsonzero()

Report.report("thompsonzero", thompsonzero, experiment_name)
