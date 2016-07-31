import json

def report(algorithm, obj, experiment_name):
	save_dir = "../../Misc/node_results/" + experiment_name + "/"
	if("coin" in algorithm):
		result_dict = {"spread":obj.spread, "regret":obj.regret, "under_exps":obj.under_exps, "squared_error":obj.squared_error}
	else:
		result_dict = {"spread":obj.spread, "regret":obj.regret, "squared_error":obj.squared_error}
	json.dump(result_dict, open(save_dir+algorithm+"_results.json", 'w'))
