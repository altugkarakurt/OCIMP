import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import numpy as np
from scipy.signal import butter, lfilter, freqz

font = {'weight' : 'semibold',
        'size'   : 15}
axisfont = {'size':'16', 'weight':'semibold'}
mpl.rc('font', **font)
mpl.rcParams['lines.linewidth'] = 2
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lpf(data, cutoff=0.5, fs=10, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

if(len(sys.argv) > 1):
	methodlist = sys.argv[2:]
else:
	print("Indicate the Methods")
	sys.exit(1)
experiment = sys.argv[1]
dicts = [json.load(open(("active_results/" + experiment +"/"+method+"_results.json"), "r")) for method in methodlist]

labels = {"coin": "COIN+", "coinhd":"COIN-HD", "coinrandom": "COIN", "oim": "CB+MLE", "thompson": "Thompson", "thompsong": "ThompsonG", "maxdegree": "High Degree", "pureexploitation": "Pure Exploitation"}


### RAW REGRET
"""
raw_regrets = [res_dict["regret"] for res_dict in dicts]
for idx, method in enumerate(methodlist):
	filtered = np.concatenate((np.array(raw_regrets[idx][:20]), lpf(raw_regrets[idx])[20:]))
	plt.plot(filtered, label=labels[method])
plt.grid()
plt.xlim([0, len(filtered)])
plt.ylim([-150,1500])
plt.legend()
plt.ylabel("Raw Regret")
plt.xlabel("Epochs")
plt.show()
"""

### AVERAGE REGRET
avg_regrets = [[sum(res_dict["regret"][:i+1])/(i+1) for i, _ in enumerate(res_dict["regret"])] for res_dict in dicts]

for idx, method in enumerate(methodlist):
	plt.plot(avg_regrets[idx], label=labels[method])
#plt.title(experiment)
plt.ylim([0,1400])
plt.xlim([0, len(avg_regrets[idx])])
plt.legend()
plt.grid()
plt.ylabel("Average Regret", **axisfont)
plt.xlabel("Epochs", **axisfont)
plt.show()

"""
### MA REGRET 50
ma = 50
ma_regrets = [[sum(res_dict["regret"][i+1-ma:i+1])/(ma) for i, _ in enumerate(res_dict["regret"])] for res_dict in dicts]

for idx, method in enumerate(methodlist):
	plt.plot(ma_regrets[idx][ma:], label=method)
plt.title(experiment)
plt.legend()
plt.ylabel("Moving Average Regret %d" % (ma))
plt.xlabel("Rounds")
plt.show()
"""

### L2 ERROR
sq_errors = [res_dict["squared_error"] for res_dict in dicts]
for idx, method in enumerate(methodlist):
	if(method != "maxdegree"):
		#plt.plot(sq_errors[idx], label=labels[method])
		filtered = np.concatenate((np.array(sq_errors[idx][:20]), lpf(sq_errors[idx])[20:]))
		plt.plot(filtered, label=labels[method])
#plt.title(experiment)
plt.xlim([0, len(filtered)])
plt.ylim([55, 87])
plt.legend()
plt.grid()
plt.ylabel("L2-Error", **axisfont)
plt.xlabel("Epochs", **axisfont)
plt.show()

"""
### SPREAD
spreads = [res_dict["spread"] for res_dict in dicts]
for idx, method in enumerate(methodlist):
	plt.plot(spreads[idx], label=method)
plt.title(experiment)
plt.legend()
plt.ylabel("Spread")
plt.xlabel("Rounds")
plt.show()

### UNDER_EXPS
for idx, method in enumerate(methodlist):
	if("coin" in method):
		plt.plot(dicts[idx]["under_exps"], label=method)
plt.title(experiment)
plt.legend()
plt.ylabel("Under Explored Nodes")
plt.xlabel("Rounds")
plt.show()
"""
