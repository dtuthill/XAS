import numpy as np
import h5py
import pickle
import sys
from mpi4py import MPI

sys.path.append('/cds/home/d/dtuthill/LY72_Spooktroscopy/')
from ly72spook_scripts import *

sys.path.append('/cds/home/j/jcryan/X400/')
from Time2Energy import T2E


# For parallelization
comm = MPI.COMM_WORLD
rank = comm.Get_rank() # which core is script being run on
size = comm.Get_size() # no. of CPUs being used
print('Connected to core {} of {}'.format(rank, size)) 

#load previously saved data

# with open('/cds/home/d/dtuthill/LY72_Spooktroscopy/runs_delay.pkl','rb') as f:
#     runs_delay = pickle.load(f)

#load file and delay calibrations
runs = np.genfromtxt('/cds/home/d/dtuthill/LY72_Spooktroscopy/lab_time_runs.csv',delimiter=',',names=True)

files = [str(run_set[0].astype(int)) + '-' + str(run_set[1].astype(int)) for run_set in runs]

runs_input = [np.arange(run_set[0].astype(int),run_set[1].astype(int)+1) for run_set in runs]
            
dlys = dly_convert(runs_input,import_table=True)

runs_delay = dict(zip(files,runs_input))
files_dlys = dict(zip(files,dlys))

#load transmission function of MBES
t_func = np.load('/cds/home/d/dtuthill/LY72_Spooktroscopy/transmission_functions/pump_probe_t_func.npy')

# determine the workload of each rank
workloads = [ len(runs_delay) // size for i in range(size) ]
for i in range( len(runs_delay) % size ):
    workloads[i] += 1

#######################################

expt = 'ly7220'

A_coeff = make_T2E_coeff(400,outer=False)

#decide with rank does which delays
if rank == 0:
    ranks_delay = list(runs_delay.keys())[0:np.cumsum(workloads)[rank]]
else:
    ranks_delay = list(runs_delay.keys())[np.cumsum(workloads)[rank-1]:np.cumsum(workloads)[rank]]

#loop through this ranks delays
for run_idx,delay in enumerate(ranks_delay):
    
    #initialize arrays
    raw = np.empty((0,1024))
    electrons = np.empty((0,3200))
    e_pump = np.empty(0)
    e_probe = np.empty(0)
    g_ints = np.empty(0)
    g_ints_err = np.empty(0)
    pump_mask = np.empty(0)
    probe_mask = np.empty(0)
    fzp_mask = np.empty(0)
    
    #loop over runs in each delay
    for run in runs_delay[delay]:
        
        print(delay,run)
        
        #load previous preprocessed MBES data and diagnostics
        gmd_path = '/cds/data/psdm/tmo/tmo%s/scratch/preproc/v7/run%d.h5' %(expt, run)

        gmd_file = h5py.File(gmd_path, 'r')
        gmd = np.array(gmd_file['gmd_energy'])
        xgmd = np.array(gmd_file['xgmd_energy'])
        hits = np.array(gmd_file['pks_MBES_inner_t_scp'])
        hits[hits == -9.99900000e+03] = np.nan
        gmd_file.close()

        #load spectrometer data
        fzp_path = '/cds/data/psdm/tmo/tmo{}/scratch/preproc/v8_fzp_only/no_y_sum/no_rebin/run{}.h5'.format('ly7220',run)
        fzp_file = h5py.File(fzp_path, 'r')

        raw_run = np.array(fzp_file['fzp_no_filter'])
        fzp_run = np.array(fzp_file['fzp_proc_masked'])
        
        g_ints_run = np.array(fzp_file['g_ints'])
        g_ints_err_run = np.array(fzp_file['g_ints_err'])
        fzp_file.close()

        #calcualte photon energy COM for each shot
        _, fzp_com_run= analyze_fzp(fzp_run,dcom=400,bgs=False)
        
        #deconvolve pump and probe energy from xgmd and gmd
        e_pump_run,e_probe_run = xgmd_gmd_cross(xgmd,gmd,pix2eV(fzp_com_run))
        
        #filter out shots with inf, nan values, and bad spectrometer amplitudes
        probe_filt = (e_probe_run != np.inf) & (e_probe_run != -np.inf) & (~np.isnan(e_probe_run))
        pump_filt = (e_pump_run != np.inf) & (e_pump_run != -np.inf) & (~np.isnan(e_pump_run))
        fzp_filt = (g_ints_run>10)*(g_ints_run<5e3)*(g_ints_run>10*g_ints_err_run)
        
        #do not actually use filter
        mask = np.ones_like(probe_filt,dtype=bool)
        #mask = probe_filt & pump_filt & fzp_filt
        
        #bin electron times-of-flights for each shot
        counts,bins = hist_laxis(hits[mask],n_bins=int(12/1e-3),range_limits=(0,12))
        
        #convert electron TOF to KE
        bins_KE,counts_KE,_ = T2E(bins[:-2],counts[:,:-1]/t_func[1],A_coeff,[0, 800, 800*4])
        
        #add data to delay array
        electrons = np.append(counts_KE,electrons,axis=0)
        raw = np.append(raw_run[mask],raw,axis=0)
        e_pump = np.append(e_pump_run[mask],e_pump)
        e_probe = np.append(e_probe_run[mask],e_probe)
        g_ints = np.append(g_ints_run[mask],g_ints)
        g_ints_err = np.append(g_ints_err_run[mask],g_ints_err)
        pump_mask = np.append(pump_filt[mask],pump_mask)
        probe_mask = np.append(probe_filt[mask],probe_mask)
        fzp_mask = np.append(fzp_filt[mask],fzp_mask)
        
    print()
        
    #bucket electron counts 
    low_idx  = np.abs(bins_KE+400-420).argmin()
    high_idx = np.abs(bins_KE+400-550).argmin()
    
    A = raw
    b = electrons[:,low_idx:high_idx].sum(-1)# - e_pump*9.559600619706092
    
    hf_save = h5py.File('/cds/home/d/dtuthill/tmoly7220/scratch/dan/datasets/grouped_no_filter/{}.h5'.format(delay.replace('.','')),'w')
    hf_save.create_dataset('A', data=A)
    hf_save.create_dataset('b', data=b)
    hf_save.create_dataset('e_pump', data=e_pump)
    hf_save.create_dataset('e_probe', data=e_probe)
    hf_save.create_dataset('g_ints',data=g_ints)
    hf_save.create_dataset('g_ints_err',data=g_ints_err)
    hf_save.create_dataset('pump_mask',data=pump_mask)
    hf_save.create_dataset('probe_mask',data=probe_mask)
    hf_save.create_dataset('fzp_mask',data=fzp_mask)
    # hf_save.create_dataset('delay',data=files_dlys[delay])
    hf_save.create_dataset('runs_input',data=runs_delay[delay])
    hf_save.close()
    
    print('Dataset {} saved succesfully'.format(delay.replace('.','')))