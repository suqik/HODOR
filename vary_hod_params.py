import os
import time
import warnings
warnings.filterwarnings('ignore')
import sys
import numpy as np
from halotools.sim_manager import UserSuppliedHaloCatalog

from WLutil.catalog import Tracer
from WLutil.base.utils import read_bigfile_header
from WLutil.measure import get_proj_2pcf
from WLutil.visualize import simple_plot

from mods_hod import *

import pymultinest as pmn
import configparser
import json

### load reference data & Covs
tmp = np.loadtxt("reference/BOSS_dr12/data/wp/galaxy_DR12v5_CMASSLOWZTOT_North.proj")
rp_ref = tmp[:,0]
ref_data_vector = tmp[:,-1]

cov=np.loadtxt("reference/BOSS_dr12/mocks/patchy_mock/wp/Patchy_mock_1000.Cov")
ref_err = np.sqrt(np.diag(cov))
### load reference DONE 

wdir="/home/suchen/Program/SBI_Void/simulator/HODOR/"
config_file=wdir+"configs/config_test_5param.ini"

config = configparser.ConfigParser()
config.read(config_file)

parameters_names = tuple(map(str, config.get('params', 'model_params_names').split(', ')))

prior_params={}
for p in parameters_names:
	prior_params[p] = list(map(float, config.get('priors', p).split(', ')))
      
vary_param_label = 2
nvary_param = 10
vary_param = np.linspace(prior_params[parameters_names[vary_param_label]][0], prior_params[parameters_names[vary_param_label]][-1], nvary_param)
model_vec = []

for parami in vary_param:
    param_list = [parami, 0.198, 13.97, 16.06, 0.802, 0.93]

    ###### my load catalog
    halo_cat_list=[]
    halo_files=[]

    header = read_bigfile_header("../catalog/calibration/jiajun/N512/a_0.7692/", dataset='RFOF/', header='Header')
    redshift = 1./header['ScalingFactor'][0] - 1
    boxsize = header['BoxSize'][0]
    particle_mass = header['MassTable'][1]*1e10

    halo = Tracer()
    halo.load_from_bigfile("../catalog/calibration/jiajun/N512/a_0.7692/", dataset='RFOF/', header='Header')

    halo_cat=UserSuppliedHaloCatalog(
        redshift=redshift,
        Lbox=boxsize,
        particle_mass=particle_mass,
        halo_upid=halo.pids,
        halo_x=halo.pos[:,0],
        halo_y=halo.pos[:,1],
        halo_z=halo.pos[:,2],
        halo_vx=halo.vel[:,0],
        halo_vy=halo.vel[:,1],
        halo_vz=halo.vel[:,2],
        halo_id=halo.ids,
        halo_rvir=halo.Radius,
        halo_mvir=halo.Mass*1e10,
        halo_nfw_conc=halo.nfw_c,  ### concentration of NFW
        halo_hostid=halo.ids
    )

    halo_cat_list.append(halo_cat)
    halo_files.append("dummy.cat")   ### give some *arbitary* names, useless

    config_file="configs/config_test_5param.ini"
    pyfcfc_conf="configs/pyfcfc_wp.conf"
    model_instance=ModelClass(config_file=config_file, halo_files=halo_files, halo_cat_list=halo_cat_list)

    num_dens_gal=3.5e-3  ### galaxy density
    dict_of_gsamples = model_instance.populate_mock(param_list, num_dens_gal)

    tmp = []
    rcat_rng = np.random.default_rng(seed=1000)
    for key in dict_of_gsamples.keys():
        x_c, y_c, z_c = dict_of_gsamples[key]["x"], dict_of_gsamples[key]["y"], dict_of_gsamples[key]["z_rsd"]
        x_c = (x_c + boxsize) % boxsize
        y_c = (y_c + boxsize) % boxsize
        z_c = (z_c + boxsize) % boxsize

        xyz_gal = np.array([x_c, y_c, z_c]).T.astype(np.float32)

        rp, wp = get_proj_2pcf(xyz_gal, boxsize=boxsize, rand1=rcat_rng.random((len(xyz_gal)*10, 3))*boxsize, conf=pyfcfc_conf, min_sep=0.5, max_sep=50, nbins=14, min_pi=0.15, max_pi=80, npbins=100,)
        tmp.append(wp)

    model_vec.append(tmp[0]*rp)
    # print(model_vec)

legends = [parameters_names[vary_param_label]+"={:.2f}".format(vary_param[i]) for i in range(nvary_param)]
# print(model_vec[0])
simple_plot([rp]*nvary_param, model_vec, legends=legends, scale='semilogx',\
             show=False, savepath=f"hod_param_{parameters_names[vary_param_label]}_tests.png")
