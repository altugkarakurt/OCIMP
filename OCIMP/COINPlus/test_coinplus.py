from COINPlus import COINPlus
import sys
sys.path.append("..")
import json
from os.path import isfile, join
from numpy.random import binomial
import numpy as np

iscontextual = True
epochs = 2000
ks = [100, 50, 25]

for k in ks:				
	pathway = ("../../Misc/deneme/coinplus__probs1__synth1__k%d__.json" % (k))
	print(pathway)

	# Experiment is already done
	if(isfile(pathway)):
		continue

	obj = COINPlus(k, "synth.txt", epochs, iscontextual)
	obj()

	result_dict = {"spread":obj.spread, "regret":obj.regret, "l2_error":obj.l2_error}
	json.dump(result_dict, open(pathway, 'w'))
