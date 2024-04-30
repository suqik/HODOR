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

######################### Loading Cov_Matrix, Ref_Data_Vector ############################
tmp = np.loadtxt("reference/BOSS_dr12/data/wp/galaxy_DR12v5_CMASSLOWZTOT_North.fits.proj")
rp_ref = tmp[:,0]
ref_data_vector = tmp[:,-1]

cov=np.loadtxt("reference/BOSS_dr12/mocks/patchy_mock/wp/Patchy_mock_2048.Cov")
ref_err = np.sqrt(np.diag(cov))
### load reference DONE 

mn_opdir = sys.argv[1]
cattype = mn_opdir.split('_')[-1].strip('/')

parameters = json.load(open(mn_opdir+"params.json", "r"))
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
param_dict = {}
for i,p in enumerate(parameters):
    param_dict[p] = param_list[i]

# get last modification time of chain file
fmtctime = time.localtime(os.path.getctime(mn_opdir+"params.json"))
fmtmtime = time.localtime(os.path.getmtime(mn_opdir+".txt"))
HMstr = time.strftime("%H%M", fmtmtime)
cDint = int(time.strftime("%d", fmtctime))
mDint = int(time.strftime("%d", fmtmtime))
Dint  = mDint - cDint # based on the day Apr. 24rd

################################## Loading HALO catalog ##################################
wdir="/home/suchen/Program/SBI_Void/simulator/HODOR/"

halo_cat_list=[]
halo_files=[]
halo = Tracer()

####### load FastPM catalog
if cattype == 'fpm':
    header = read_bigfile_header(wdir+"../catalog/calibration/jiajun/N512/a_0.7692/", dataset='RFOF/', header='Header')
    redshift = 1./header['ScalingFactor'][0] - 1
    boxsize = header['BoxSize'][0]
    particle_mass = header['MassTable'][1]*1e10 
    halo.load_from_bigfile(wdir+"../catalog/calibration/jiajun/N1024/a_0.7692/", dataset='RFOF/', header='Header')
    halo.Mass = halo.Mass*1e10 # Msun/h

####### load Nbody catalog
if cattype == 'nbd':
    halo.load_from_ascii(wdir+"../catalog/Nbody/jiajun/snapshot_011.z0.300.AHF_halos",\
                        ids=0, 
                        pos=[5,7],
                        vel=[8,10],
                        pids=1, 
                        Radius=11, 
                        Mass=3,
                        nfw_c=42)
    halo.kpc2Mpc()
    redshift = 0.3
    boxsize = 400.0
    particle_mass = 4.17684421*1e10

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
    halo_mvir=halo.Mass,
    halo_nfw_conc=halo.nfw_c,  ### concentration of NFW
    halo_hostid=halo.ids
)

halo_cat_list.append(halo_cat)
halo_files.append("dummy.cat")   ### give some *arbitary* names, useless

config_file="configs/config_test_5param.ini"
pyfcfc_conf="configs/pyfcfc_wp.conf"

#################### Apply HOD ########################
model_instance=ModelClass(config_file=config_file, halo_files=halo_files, halo_cat_list=halo_cat_list)

num_dens_gal=3.5e-4  ### galaxy density
dict_of_gsamples = model_instance.populate_mock(param_list, num_dens_gal)

total_gal_num=0
for key in dict_of_gsamples.keys():
    total_gal_num+=len(dict_of_gsamples[key])

ave_dens_model=total_gal_num/len(dict_of_gsamples)/boxsize/boxsize/boxsize

tmp = []
rcat_rng = np.random.default_rng(seed=1000)
for key in dict_of_gsamples.keys():
    x_c, y_c, z_c = dict_of_gsamples[key]["x"], dict_of_gsamples[key]["y"], dict_of_gsamples[key]["z_rsd"]
    x_c = (x_c + boxsize) % boxsize
    y_c = (y_c + boxsize) % boxsize
    z_c = (z_c + boxsize) % boxsize

    xyz_gal = np.array([x_c, y_c, z_c]).T.astype(np.float32)

    rp, wp = get_proj_2pcf(xyz_gal, boxsize=boxsize, rand1=rcat_rng.random((len(xyz_gal)*10, 3))*boxsize, conf=pyfcfc_conf, min_sep=0.5, max_sep=50, nbins=14, min_pi=0, max_pi=80, npbins=40,)
    tmp.append(wp)

model_vector = np.mean(np.asarray(tmp), axis=0)
model_err    = np.std(np.asarray(tmp), axis=0)

####################### Save data & Plots ##########################
if not os.path.isdir(f"results_{cattype}/"):
    os.mkdir(f"results_{cattype}/")

##### Save data & error
if not os.path.isdir(f"results_{cattype}/proj"):
    os.mkdir(f"results_{cattype}/proj")

str_param = str(param_list).strip('[]')
f = open(f"results_{cattype}/proj/hod_bestfit_{Dint}_{HMstr}.proj", "w+", encoding='utf-8')
f.write(f"# parameters: {str_param}\n")
f.write("# average number density: {:.4e}\n".format(ave_dens_model))
np.savetxt(f, np.c_[rp, model_vector, model_err])
f.close()

##### Plot N(M) figures 
if not os.path.isdir(f"results_{cattype}/NM_fig"):
    os.mkdir(f"results_{cattype}/NM_fig")

ctr = hod_models.MWCens(redshift=0.3)
ctr.param_dict = param_dict
Nctr = ctr.mean_occupation(prim_haloprop=halo.Mass)

sat = hod_models.MWSats(redshift=0.3)
sat.param_dict = param_dict
Nsat = sat.mean_occupation(prim_haloprop=halo.Mass)
simple_plot([np.log10(halo.Mass)]*3, [Nctr, Nsat, Nctr+Nsat], 
            xlim=[12.5, 15.5], ylim=[0.1, 30], xlabel=r'$M_h$', ylabel=r'$N(M)$',
            clist=['r', 'b', 'k'], lslist=['-', '--', ':'],
             legends=['Cen', 'Sat', 'All'], scale='semilogy',
               show=False, savepath=f"results_{cattype}/NM_fig/hod_bestfit_{Dint}_{HMstr}.png")

##### Plot Comparison figures 
if not os.path.isdir(f"results_{cattype}/cmp_fig"):
    os.mkdir(f"results_{cattype}/cmp_fig")
hod_result = np.loadtxt(f"results_{cattype}/proj/hod_bestfit_{Dint}_{HMstr}.proj")
cov_hod = np.diag(hod_result[:,2]**2)

cut = 4
has_mCov = False
vline = (rp_ref[cut-1]*np.sqrt(rp_ref[2]/rp_ref[1]))

diff_vec = hod_result[cut:,1] - ref_data_vector[cut:]
if has_mCov:
    cov_tot = cov[cut:,cut:] + cov_hod[cut:,cut:]
else:
    cov_tot = cov[cut:,cut:]
icov_tot = np.linalg.inv(cov_tot)

chi2 = diff_vec@icov_tot@diff_vec

simple_plot([hod_result[:,0], rp_ref, [vline, vline]], [hod_result[:,1]*hod_result[:,0], ref_data_vector*rp_ref, [0,1000]], yerrlist=[hod_result[:,2]*hod_result[:,0], ref_err*rp_ref, None],\
                legends=['FPM_HOD chi2={:.2f}'.format(chi2),'BOSS',None], clist=[None,None,'k'], lslist=['-','-','--'], 
                 ylim=[50,350], xlabel='rp [Mpc/h]', ylabel='Rp x wp', scale='semilogx',\
                    show=False, savepath=f"results_{cattype}/cmp_fig/hod_bestfit_proj_{Dint}_{HMstr}.png")
