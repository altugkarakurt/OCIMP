from OIMMulti import OIMMulti
import sys
sys.path.append("..")
import Report

graph_file = "subnethept.txt"
iscontextual = True
experiment_name = "contextual_"
experiment_name += graph_file[:-4]
rounds = 5000
seed_size = 50

oimmulti = OIMMulti(seed_size, graph_file, rounds, iscontextual)
oimmulti()

Report.report("oimmulti", oimmulti, experiment_name)
