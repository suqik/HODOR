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
import json

### load reference data & Covs
tmp = np.loadtxt("reference/BOSS_dr12/data/wp/galaxy_DR12v5_CMASSLOWZTOT_North.proj")
rp_ref = tmp[:,0]
ref_data_vector = tmp[:,-1]

cov=np.loadtxt("reference/BOSS_dr12/mocks/patchy_mock/wp/Patchy_mock_1000.Cov")
ref_err = np.sqrt(np.diag(cov))
### load reference DONE 

mn_opdir = sys.argv[1]

# parameters = json.load(open(mn_opdir+"params.json", "r"))
# n_params = len(parameters)

# a = pmn.Analyzer(n_params = n_params, outputfiles_basename = mn_opdir)
# s = a.get_stats()


# param_list = []
# for p, m in zip(parameters, s['marginals']):
# 	lo, hi = m['1sigma']
# 	med = m['median']
# 	sigma = (hi - lo) / 2
# 	param_list.append(med)
# param_list = np.asarray(param_list)

# param_list = [12.56, 0.198, 13.97, 16.06, 0.802, 0.93]
# param_list[-1] = 0.0
# print(param_list)

### choose the param that have the maximum likelihood
full_chain = np.loadtxt(mn_opdir+".txt")
points = full_chain[:,2:]
m2loglike = (full_chain[:,1])
min_label = np.argmin(m2loglike) # -2*loglikelihood (ln, actually), i.e., chi2
param_list = points[min_label]

# get last modification time of chain file
fmttime = time.localtime(os.path.getmtime(mn_opdir+".txt"))
HMstr = time.strftime("%H%M", fmttime)
Dint = int(time.strftime("%d", fmttime))
Dint -= 24 # based on the day Apr. 24rd

if Dint == 0:
    Dstr = ""
else:
    Dstr = f"_{Dint}"

###### my load catalog
# halo_cat_list=[]
# halo_files=[]

# header = read_bigfile_header("../catalog/calibration/jiajun/N512/a_0.7692/", dataset='RFOF/', header='Header')
# redshift = 1./header['ScalingFactor'][0] - 1
# boxsize = header['BoxSize'][0]
# particle_mass = header['MassTable'][1]*1e10

# halo = Tracer()
# halo.load_from_bigfile("../catalog/calibration/jiajun/N512/a_0.7692/", dataset='RFOF/', header='Header')

# halo_cat=UserSuppliedHaloCatalog(
#     redshift=redshift,
#     Lbox=boxsize,
#     particle_mass=particle_mass,
#     halo_upid=halo.pids,
#     halo_x=halo.pos[:,0],
#     halo_y=halo.pos[:,1],
#     halo_z=halo.pos[:,2],
#     halo_vx=halo.vel[:,0],
#     halo_vy=halo.vel[:,1],
#     halo_vz=halo.vel[:,2],
#     halo_id=halo.ids,
#     halo_rvir=halo.Radius,
#     halo_mvir=halo.Mass*1e10,
#     halo_nfw_conc=halo.nfw_c,  ### concentration of NFW
#     halo_hostid=halo.ids
# )

# halo_cat_list.append(halo_cat)
# halo_files.append("dummy.cat")   ### give some *arbitary* names, useless

# config_file="configs/config_test_5param.ini"
# pyfcfc_conf="configs/pyfcfc_wp.conf"
# model_instance=ModelClass(config_file=config_file, halo_files=halo_files, halo_cat_list=halo_cat_list)

# num_dens_gal=3.5e-3  ### galaxy density
# dict_of_gsamples = model_instance.populate_mock(param_list, num_dens_gal)

# tmp = []
# rcat_rng = np.random.default_rng(seed=1000)
# for key in dict_of_gsamples.keys():
#     x_c, y_c, z_c = dict_of_gsamples[key]["x"], dict_of_gsamples[key]["y"], dict_of_gsamples[key]["z_rsd"]
#     x_c = (x_c + boxsize) % boxsize
#     y_c = (y_c + boxsize) % boxsize
#     z_c = (z_c + boxsize) % boxsize

#     xyz_gal = np.array([x_c, y_c, z_c]).T.astype(np.float32)

#     rp, wp = get_proj_2pcf(xyz_gal, boxsize=boxsize, rand1=rcat_rng.random((len(xyz_gal)*10, 3))*boxsize, conf=pyfcfc_conf, min_sep=0.5, max_sep=50, nbins=14, min_pi=0.15, max_pi=80, npbins=100,)
#     tmp.append(wp)

# model_vector = np.mean(np.asarray(tmp), axis=0)
# model_err    = np.std(np.asarray(tmp), axis=0)

# str_param = str(param_list).strip('[]')
# f = open(f"results/proj/hod_bestfit{Dstr}_{HMstr}.proj", "w+", encoding='utf-8')
# f.write(f"# parameters: {str_param}\n")
# np.savetxt(f, np.c_[rp, model_vector, model_err])
# f.close()

hod_result = np.loadtxt(f"results/proj/hod_bestfit{Dstr}_{HMstr}.proj")
cov_hod = np.diag(hod_result[:,2])
cov_tot = cov + cov_hod
icov_tot = np.linalg.inv(cov_tot)

diff_vec = hod_result[:,1] - ref_data_vector

chi2 = diff_vec@icov_tot@diff_vec
# chi2 = np.dot(diff_vec.T, np.dot(icov_tot, diff_vec))
# print(chi2[0])

simple_plot([hod_result[:,0], rp_ref], [hod_result[:,1]*hod_result[:,0], ref_data_vector*rp_ref], yerrlist=[hod_result[:,2]*hod_result[:,0], ref_err*rp_ref],\
                legends=['FPM_HOD chi2={:.2f}'.format(chi2),'BOSS'], xlabel='rp [Mpc/h]', ylabel='Rp x wp', scale='semilogx',\
                    show=False, savepath=f"results/cmp_fig/hod_bestfit_proj{Dstr}_{HMstr}.png")
