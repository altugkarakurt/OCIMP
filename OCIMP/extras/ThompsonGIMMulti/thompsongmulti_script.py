from ThompsonGIMMulti import ThompsonGIMMulti
import sys
sys.path.append("..")
import Report

graph_file = "subnethept.txt"
iscontextual = True
experiment_name = "contextual_"
experiment_name += graph_file[:-4]
rounds = 5000
seed_size = 50

thompsongmulti = ThompsonGIMMulti(seed_size, graph_file, rounds, iscontextual)
thompsongmulti()

Report.report("thompsongmulti", thompsongmulti, experiment_name)
