import os, sys
import time
import numpy as np
import configparser
import json
from getdist import MCSamples, plots

base = sys.argv[1]
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
                     ranges=prior_params, loglikes=loglikes) #

# get last modification time of chain file
fmttime = time.localtime(os.path.getmtime(base+".txt"))
HMstr = time.strftime("%H%M", fmttime)
Dint = int(time.strftime("%d", fmttime))
Dint -= 24

if Dint == 0:
    Dstr = ""
else:
    Dstr = f"_{Dint}"

g = plots.get_subplot_plotter(subplot_size=1)
g.triangle_plot([samples], 
                filled=True, 
                contour_lws=1.2 , 
                )
g.export(f'results/contour/hod_5params_wRSD{Dstr}_{HMstr}_w_prior.png',dpi=400)
