import json
import numpy as np
import sys

method = sys.argv[1]
count = int(sys.argv[2])
experiment = sys.argv[3] if(len(sys.argv) > 3) else "contextual_nethept"

methodlist = [method+str(i) for i in range(1,count+1)]
dicts = [json.load(open(("results/" + experiment + "/"+method+"_results.json"), "r"))
		 for method in methodlist]
avg_dict = dict()

for key in dicts[0].keys():
	attr_list = [d[key] for d in dicts]
	avg_list = np.sum(attr_list, axis=0) / len(dicts)
	avg_dict[key] = avg_list.tolist()

json.dump(avg_dict, open(("results/" + experiment + "/averages/"+method+"_avg.json"), "w"))