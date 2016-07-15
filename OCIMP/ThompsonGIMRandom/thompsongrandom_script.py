from ThompsonGIMRandom import ThompsonGIMRandom
import sys
sys.path.append("..")
import Report

graph_file = "nethept.txt"
experiment_name = "noncontextual_" + graph_file[:-4]
rounds = 1250
seed_size = 50

thompsongrandom = ThompsonGIMRandom(seed_size, graph_file, rounds)
thompsongrandom()

Report.report("thompsongrandom", thompsongrandom, experiment_name)
