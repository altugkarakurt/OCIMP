from COINPlus import COINPlus
import sys
sys.path.append("..")
import Report
from os.path import isfile, join

seed_size = 50
cost = 0.1

for iscontextual in [True, False]:
	for graph_file in ["subnethept.txt", "nethept.txt"]:
		if((graph_file == "subnethept.txt") and (iscontextual)):
			epochs = 5000
		elif((graph_file == "nethept.txt") and (iscontextual)):
			epochs = 5000
		elif((graph_file == "subnethept.txt") and (not iscontextual)):
			epochs = 1250
		elif((graph_file == "nethept.txt") and (not iscontextual)):
			epochs = 2500
		experiment_name = "contextual_" if(iscontextual) else "noncontextual_"
		experiment_name += graph_file[:-4]

		pathway = join("../../Misc/active_results", experiment_name, "coinzero_results.json")

		# Experiment is already done
		if(isfile(pathway)):
			continue

		coinplus = COINPlus(seed_size, graph_file, epochs, iscontextual, cost)
		coinplus()

		Report.report("coinplus", coinplus, experiment_name)
