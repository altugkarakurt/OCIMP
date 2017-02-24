from COINPlus import COINPlus
import sys
sys.path.append("..")
import Report
from os.path import isfile, join

seed_size = 50
iscontextual = True

for graph_file in ["nethept.txt", "subnethept.txt"]:
	if(graph_file == "subnethept.txt"):
		epochs = 5000
	elif(graph_file == "nethept.txt"):
		epochs = 5000
	experiment_name = "contextual_" + graph_file[:-4]

	pathway = join("../../Misc/node_results/sigmoid", experiment_name, "coinplus_results.json")

	# Experiment is already done
	if(isfile(pathway)):
		continue
	print(pathway)
	coinplus = COINPlus(seed_size, graph_file, epochs, iscontextual)
	coinplus()

	Report.report("coinplus", coinplus, experiment_name)
