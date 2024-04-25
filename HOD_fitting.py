import os
import warnings
warnings.filterwarnings('ignore')

import configparser
import json
import numpy as np

from halotools.sim_manager import UserSuppliedHaloCatalog
from mods_hod import *

from WLutil.catalog import Tracer
from WLutil.base.utils import read_bigfile_header
from WLutil.measure import get_auto_2pcf, get_proj_2pcf

import pymultinest as pmn

wdir="/home/suchen/Program/SBI_Void/simulator/HODOR/"
config_file=wdir+"configs/config_test_5param.ini"
pyfcfc_conf=wdir+"configs/pyfcfc_wp.conf"

config = configparser.ConfigParser()
config.read(config_file)

num_params = config['params'].getint('num_params')
parameters_names = tuple(map(str, config.get('params', 'model_params_names').split(', ')))

prior_params={}
for p in parameters_names:
	prior_params[p] = tuple(map(float, config.get('priors', p).split(', ')))

######################### Loading Cov_Matrix, Ref_Data_Vector ############################
### BOSS projected correlation functions
### rp numbins and range; see BOSS HOD paper
data_vec = []

tmp = np.loadtxt(wdir+"reference/BOSS_dr12/data/wp/galaxy_DR12v5_CMASSLOWZTOT_North.fits.proj")
rp_ref = tmp[:,0]
ref_data_vector = tmp[:,-1]

cov=np.loadtxt(wdir+"reference/BOSS_dr12/mocks/patchy_mock/wp/Patchy_mock_2048.Cov")
icov = np.linalg.inv(cov)

####### my load catalog
halo_cat_list=[]
halo_files=[]

header = read_bigfile_header(wdir+"../catalog/calibration/jiajun/N512/a_0.7692/", dataset='RFOF/', header='Header')
redshift = 1./header['ScalingFactor'][0] - 1
boxsize = header['BoxSize'][0]
particle_mass = header['MassTable'][1]*1e10

halo = Tracer()
halo.load_from_bigfile(wdir+"../catalog/calibration/jiajun/N512/a_0.7692/", dataset='RFOF/', header='Header')

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

### make 10 copies of the same halo catalogue, 
### generate 10 HOD realizations,
### take an average of their correlation functions,
### to decrease cosmic variance

# ################### Initialize HOD Models #####################

model_instance=ModelClass(config_file=config_file, halo_files=halo_files, halo_cat_list=halo_cat_list)

num_dens_gal=3.5e-3  ### galaxy density

rcat_rng = np.random.default_rng(seed=1000)

def loglike(cube, ndim, nparams):
### ndim and nparams are not recalled in the function, but necessary.

    param=cube[0: num_params]
    
    if (((10**15-param[3]*(10**param[0]))/(10**param[2]))**param[4])>1000:
        ### if N_sat for halo mass 1e15 is larger than 1000, then drop it
        return -1e101

    dict_of_gsamples = model_instance.populate_mock(param, num_dens_gal)
    ### generate mock galaxy catalogues

    total_gal_num=0
    for key in dict_of_gsamples.keys():
        total_gal_num+=len(dict_of_gsamples[key])

    ave_dens_model=total_gal_num/len(dict_of_gsamples)/boxsize/boxsize/boxsize

    if np.abs(ave_dens_model - num_dens_gal) > num_dens_gal * 0.1:
        ### if galaxy number density is too low or too high, then drop it
        return -1e101

    tmp = []
    for key in dict_of_gsamples.keys():
        x_c, y_c, z_c = dict_of_gsamples[key]["x"], dict_of_gsamples[key]["y"], dict_of_gsamples[key]["z_rsd"]
        x_c = (x_c + boxsize) % boxsize
        y_c = (y_c + boxsize) % boxsize
        z_c = (z_c + boxsize) % boxsize

        xyz_gal = np.array([x_c, y_c, z_c]).T.astype(np.float32)

        # s, mono, quad, hex = get_auto_2pcf(xyz_gal, boxsize=boxsize, rand1=np.random.rand(len(xyz_gal), 3)*boxsize, min_sep=3, max_sep=198, scale='linear', nbins=66)
        rp, wp = get_proj_2pcf(xyz_gal, boxsize=boxsize, rand1=rcat_rng.random((len(xyz_gal)*10, 3))*boxsize, conf=pyfcfc_conf, min_sep=0.5, max_sep=50, nbins=14, min_pi=0, max_pi=80, npbins=40,)

        # tmp.append(mono*s**2)
        tmp.append(wp)
    model_vector = np.mean(np.asarray(tmp), axis=0)

    ### here the output Xi0 and Xi2 have already been averaged over all 10 galaxy catalogues.

    diff_vector=ref_data_vector-np.array(model_vector)

    chi2=np.dot(diff_vector.T, np.dot(icov, diff_vector))
    chi2=-0.5*chi2

    if chi2 is np.nan:
        return -1e101
    else:
        return chi2

def prior(cube, ndim, nparams):
    for i in range(num_params):
        val_max=prior_params[parameters_names[i]][1]
        val_min=prior_params[parameters_names[i]][0]

        cube[i] = cube[i] * (val_max - val_min) + val_min

multinest_verbose=config["multinest"].getboolean("verbose")
multinest_tol=config["multinest"].getfloat("tol")
multinest_live_points=config["multinest"].getint("live_points")

multinest_opdir = wdir+"multinest_out/"
if not os.path.isdir(multinest_opdir):
    os.mkdir(multinest_opdir)

### save parameter name as json file
with open(multinest_opdir+"params.json", "w+") as f:
    json.dump(parameters_names, f, indent=2)

pmn.run(loglike, 
        prior, 
        num_params, 
        outputfiles_basename=multinest_opdir, 
        resume=False,
        verbose=multinest_verbose, 
        n_live_points=multinest_live_points, 
        evidence_tolerance=multinest_tol, 
        seed=79, 
        n_iter_before_update = 10,    ### update the output files every 10 interations
        importance_nested_sampling = False)