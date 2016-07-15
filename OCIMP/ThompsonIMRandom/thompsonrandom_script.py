from ThompsonIMRandom import ThompsonIMRandom
import sys
sys.path.append("..")
import Report

graph_file = "nethept.txt"
experiment_name = "noncontextual_" + graph_file[:-4]
rounds = 1250
seed_size = 50

thompsonrandom = ThompsonIMRandom(seed_size, graph_file, rounds)
thompsonrandom()

Report.report("thompsonrandom", thompsonrandom, experiment_name)
