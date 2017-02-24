from COINPlus import COINPlus
import sys
sys.path.append("..")
import Report
from os.path import isfile, join

seed_size = 50
iscontextual = True
cost = 0.1

for graph_file in ["subnethept.txt", "nethept.txt"]:
	if(graph_file == "subnethept.txt"):
		epochs = 5000
	elif(graph_file == "nethept.txt"):
		epochs = 5000
	experiment_name = "contextual_" + graph_file[:-4]

	pathway = join("../../Misc/active_results/sigmoid", experiment_name, "coinplus_results.json")

	# Experiment is already done
	if(isfile(pathway)):
		continue
	print(pathway)
	coinplus = COINPlus(seed_size, graph_file, epochs, iscontextual, cost)
	coinplus()

	Report.report("coinplus", coinplus, experiment_name)

