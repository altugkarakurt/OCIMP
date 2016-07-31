from COINHDZero import COINHDZero
import sys
sys.path.append("..")
import Report

seed_size = 50
cost = 0.1

for iscontextual in [True, False]:
	for graph_file in ["subnethept.txt", "nethept.txt"]:
		if((graph_file == "subnethept.txt") and (iscontextual)):
			rounds = 5000
		elif((graph_file == "nethept.txt") and (iscontextual)):
			rounds = 5000
		elif((graph_file == "subnethept.txt") and (not iscontextual)):
			rounds = 1250
		elif((graph_file == "nethept.txt") and (not iscontextual)):
			rounds = 2500
		experiment_name = "contextual_" if(iscontextual) else "noncontextual_"
		experiment_name += graph_file[:-4]

		pathway = join("../../Misc/active_results", experiment_name, "coinhdzero")

		# Experiment is already done
		if(isfile(pathway)):
			continue

		coinhdzero = COINHDZero(seed_size, graph_file, rounds, iscontextual, cost)
		coinhdzero()

		Report.report("coinhdzero", coinhdzero, experiment_name)
