from ThompsonGIMZero import ThompsonGIMZero
import sys
sys.path.append("..")
import Report

graph_file = "nethept.txt"
experiment_name = "noncontextual_" + graph_file[:-4]
rounds = 2500
seed_size = 50

thompsongzero = ThompsonGIMZero(seed_size, graph_file, rounds)
thompsongzero()

Report.report("thompsongzero", thompsongzero, experiment_name)
