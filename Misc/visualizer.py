import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import numpy as np

"""--------------------------------------------------------------------
The script to visualize the results. It prints out raw regret, average
regret and l2-error figures.Make sure to correct line 24 to look for
the correct one of "results", "active_results" or "node_results". Then,
you can use this script the following way:

python visualizer.py [experiment_name] [method1] [method2] ...

ex: python visualizer.py noncontextual_nethept coin coin+ thompsong
--------------------------------------------------------------------"""

### Tweaks to make matplotlib look pretty
font = {'weight' : 'semibold',
        'size'   : 15}
axisfont = {'size':'16', 'weight':'semibold'}
mpl.rc('font', **font)
mpl.rcParams['lines.linewidth'] = 2

if(len(sys.argv) > 1):
	methodlist = sys.argv[2:]
else:
	print("Indicate the Methods")
	sys.exit(1)
experiment = sys.argv[1]
dicts = [json.load(open(("results/" + experiment +"/"+method+"_results.json"), "r")) \
		for method in methodlist]

labels = {"epsilongreedy":"$\epsilon_n$-Greedy", "coin+": "COIN+", "coinhd":"COIN-HD",
		  "coin": "COIN", "oim": "CB+MLE", "thompson": "Thompson", "thompsong": "ThompsonG",
		  "highdegree": "High Degree", "pureexploitation": "Pure Exploitation"}
colors = ["b", "g", "r", "c", "m", "y", "k", "orange"]

### RAW REGRET
raw_regrets = [res_dict["regret"] for res_dict in dicts]
for idx, method in enumerate(methodlist):
	plt.plot(list(np.arange(900, 1250)), raw_regrets[idx][900:1250], colors[idx], label=labels[method])
plt.grid()
plt.legend()
plt.ylabel("Raw Regret")
plt.xlabel("Epochs")
plt.show()


### AVERAGE REGRET
avg_regrets = [[sum(res_dict["regret"][:i+1])/(i+1) for i, _ in enumerate(res_dict["regret"])] for res_dict in dicts]

for idx, method in enumerate(methodlist):
	plt.plot(list(range(5000)), avg_regrets[idx][:len(avg_regrets[0])], colors[idx],  label=labels[method])
plt.title(experiment)
plt.legend()
plt.grid()
plt.ylabel("Average Regret", **axisfont)
plt.xlabel("Epochs", **axisfont)
plt.show()

### L2 ERROR
l2_errors = [res_dict["l2_error"] for res_dict in dicts]
for idx, method in enumerate(methodlist):
	if(method != "maxdegree"):
		plt.plot(list(range(len(l2_errors[idx]))), l2_errors[idx], colors[idx], label=labels[method])
plt.title(experiment)
plt.legend()
plt.grid()
plt.ylabel("L2-Error", **axisfont)
plt.xlabel("Epochs", **axisfont)
plt.show()
