import os, sys
import time
import numpy as np
import configparser
import json
from getdist import MCSamples, plots

base = sys.argv[1]
cattype = base.split("_")[-1].strip('/')

config_file = "configs/config_test_5param.ini"

config = configparser.ConfigParser()
config.read(config_file)

num_params = config['params'].getint('num_params')
parameters_names = list(map(str, config.get('params', 'model_params_names').split(', ')))

prior_params={}
for p in parameters_names:
    prior_params[p] = list(map(float, config.get('priors', p).split(', ')))

with open(base+"params.json") as f:
    name_label = json.load(f)

tmp = np.loadtxt(base+".txt")
# weights = tmp[:,0]
loglikes = (tmp[:,1])/2 # -loglikes
samples = MCSamples(samples=tmp[:,2:], names=parameters_names, labels=['logM_{cut}', '\sigma_{logM_1}', 'logM_1', name_label[3], '\\alpha'],\
                     ranges=prior_params, loglikes=loglikes) #, weights=weights
samples2 = MCSamples(samples=tmp[:,2:], names=parameters_names, labels=['logM_{cut}', '\sigma_{logM_1}', 'logM_1', name_label[3], '\\alpha'],)

# get last modification time of chain file
fmtctime = time.localtime(os.path.getctime(base+"params.json"))
fmtmtime = time.localtime(os.path.getmtime(base+".txt"))
HMstr = time.strftime("%H%M", fmtmtime)
cDint = int(time.strftime("%d", fmtctime))
mDint = int(time.strftime("%d", fmtmtime))
Dint  = mDint - cDint

g = plots.get_subplot_plotter(subplot_size=1)
g.triangle_plot([samples, samples2], 
                legend_labels=["w/ prior", "w/o prior"],
                filled=False, 
                contour_lws=1.2 , 
                )
g.export(f'results_{cattype}/contour/hod_5params_wRSD_{Dint}_{HMstr}.png',dpi=400)
