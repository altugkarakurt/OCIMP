from COINMLERandom import COINMLERandom
import sys
sys.path.append("..")
import Report

graph_file = "subnethept.txt"
contextual = True
experiment_name = "contextual_" if(iscontextual) else "noncontextual_"
experiment_name += graph_file[:-4]
rounds = 1250
seed_size = 50

coinmlerandom = COINMLERandom(seed_size, graph_file, rounds, iscontextual)
coinmlerandom()

Report.report("coinmlerandom", coinmlerandom, experiment_name)
