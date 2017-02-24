from HighDegree import HighDegree
import sys
sys.path.append("..")
import Report
from os.path import isfile, join

seed_size = 50
iscontextual = True

for graph_file in ["subnethept.txt", "nethept.txt"]:
	if(graph_file == "subnethept.txt"):
		epochs = 5000
	elif(graph_file == "nethept.txt"):
		epochs = 5000
	experiment_name = "contextual_" + graph_file[:-4]

	pathway = join("../../Misc/results/sigmoid", experiment_name, "highdegree_results.json")

	# Experiment is already done
	if(isfile(pathway)):
		continue
	print(pathway)
	highdegree = HighDegree(seed_size, graph_file, epochs, iscontextual)
	highdegree()

	Report.report("highdegree", highdegree, experiment_name)
