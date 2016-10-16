import numpy as np
from numpy.random import shuffle, choice
import networkx as nx

"""--------------------------------------------------------------------
This is the script used to generate the graph NetHEPT-. We didn't remove
any extra nodes but one can do so by changing the ratio parameter.
--------------------------------------------------------------------"""

graph_file = "nethept.txt"
ratio = 1
indegrisk = True
outdegrisk = True

edges = [(int(line.strip("\n").split("\t")[0]), 
             int(line.strip("\n").split("\t")[1])) \
             for line in open(graph_file, "r")]

dig = nx.DiGraph(edges)
to_remove = choice(dig.nodes(), int((1-ratio) * len(dig.nodes())), replace=False)
dig.remove_nodes_from(to_remove)

while(indegrisk):
	print("Risk")
	to_remove = []
	for node, deg in dig.in_degree_iter():
		if deg == 0:
			to_remove.append(node)
	if(len(to_remove) == 0):
		indegrisk = False
	"""
	for node, deg in dig.out_degree_iter():
		if deg == 0:
			to_remove.append(node)
	"""
	dig.remove_nodes_from(to_remove)

nodes = dig.nodes()
edges = dig.edges()
print("Edge: %d" % (len(edges)))
print("Node: %d" % (len(nodes)))
new_edges = [[nodes.index(edge[0]), nodes.index(edge[1])] for edge in edges]
np.savetxt("nonzeronethept.txt", new_edges, delimiter="\t", fmt=["%d", "%d"])