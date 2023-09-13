import numpy as np
import h5py
from analysis_functions import *
from dataclasses import dataclass
from tqdm.notebook import tqdm
import sys
import pickle
from scipy.optimize import curve_fit

# sys.path.append('/cds/home/i/isele/campaign/libraries/')
# import get_XAS_runs_LY72

'''
sub-classes of XAS class
'''

class normalizations:
    
    '''
    normalization constants and functiosn for data
    '''
    
    def __init__(self):
        
        #MBES transmission function
        self.t_func = np.load('/cds/home/d/dtuthill/LY72_Spooktroscopy/transmission_functions/pump_probe_t_func.npy')
        
        #YAG transmission function for FZP spectrometer
        yag_path = '/cds/home/i/isele/tmoly7220/results/Erik/spook/yag_transmission/yag_t.h5'
        yag_file = h5py.File(yag_path,'r')
        #yag_bins = np.array(yag_file['pes'])
        self.yag = np.array(yag_file['t_func'])
        yag_file.close()
        
        #Regression dilution correction matrix
        auto_mat_path = '/cds/home/i/isele/tmoly7220/results/Erik/spook/h5/fzp_dark_TtT_norm_mat_170723_1024pix.h5'
        hf = h5py.File(auto_mat_path,'r')
        self.TtT_norm = np.array(hf.get('TtT_norm'))*np.outer(self.yag, self.yag)*100 # Account for YAG transmission function
        hf.close()
        
        
class masks:
    
    '''
    mask to remove shots that are cut by the edges of the FZP spectrometer
    Creates edge mask by fitting a Gaussian, filtering background noise, and determining if zero at edges
    '''
        
    def create_edge_mask(self,files,A,fzp_eV,create=False,save=False,edge_bins=120,round_sigfig=0,save_path = '/cds/home/d/dtuthill/LY72_Spooktroscopy/xas/masks'):
        
        #Performs Gaussian fits
        
        popt = {}
        bad = {}
        
        if create:
                    
            for i,file in enumerate(tqdm(files, desc='Fitting Gaussians')):
                
                #for each dataset, fit gaussian function
                #saves gaussian fitting parameters as h5
                #note, this takes a LONG time to run (~20 hours). it is suggested to save result

                popt_delay = np.empty((A[file].shape[0],4))
                x = np.arange(fzp_eV.shape[0])
                bad_delay = np.full(A[file].shape[0], False)

                for i in range(A[file].shape[0]):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            popt_delay[i],_ = curve_fit(gauss,x,A[file][i],p0=[(A[file][i]).max(),(A[file][i]).argmax(),16/(1024/A[file].shape[-1]),0])
                    except:
                        bad_delay[i] = True

                popt[file] = popt_delay
                bad[file] = bad_delay
                
                if save:

                    hf_save = h5py.File(save_path+'/{}.h5'.format(file),'w')
                    hf_save.create_dataset('popt', data=popt_delay)
                    hf_save.create_dataset('bad', data=bad_delay)
                    hf_save.close()

                    print('Dataset {} saved succesfully'.format(file))
                
        else:
            
            for i,file in enumerate(tqdm(files,desc = 'Importing Gaussian fits')):
                
                #imports previously saved fits
    
                hf_file = h5py.File(save_path+'/{}.h5'.format(file),'r')
                popt[file] = np.array(hf_file['popt'])
                bad[file] = np.array(hf_file['bad'])
                hf_file.close()
                
        self.popt = popt
        
        #Uses gaussian fits to determine which ones are zero at both edges of the spectrometer
        #To account for noise, rounds to integer amplitude values
        #Edge bins is how far from edge the gaussian must be zero
                
        mask = {}

        for i,file in enumerate(tqdm(files,desc='Creating edge masks')):

            popt_delay = popt[file]
            bad_delay = bad[file]
    
            x = np.tile(np.arange(fzp_eV.shape[0]),(A[file].shape[0],1))
            fits = gauss(np.tile(np.arange(fzp_eV.shape[0]),(A[file].shape[0],1)),
                         popt_delay[:,0].reshape(-1,1),
                         popt_delay[:,1].reshape(-1,1),
                         popt_delay[:,2].reshape(-1,1),
                         popt_delay[:,3].reshape(-1,1)) - popt_delay[:,3].reshape(-1,1)

            mask_delay = (np.abs(np.around(fits[:,:edge_bins],round_sigfig)).sum(-1) > 1) | (np.abs(np.around(fits[:,-edge_bins:],round_sigfig)).sum(-1) > 1) | bad_delay
            mask[file] = ~mask_delay
            
            print('Delay: ',file,'% removed: ',mask_delay.sum()/len(mask_delay))
            
        self.edge_masks = mask
    
@dataclass
class params:
    #Analysis parameters
    nbins: int = 256
    pump_bins: int = 6
    pump_min: float = 0.00
    pump_max: float = 0.1
    lsmooth: float = 5.2e2
    bpmx_bins: int = 3
    bpmx_min: float = -0.2
    bpmx_max: float = 0.2
    und_filt: int = 36
    
class runs:
    
    '''
    Class of run data including:
    -delay
    -runs per delay
    -uncoverted delay (file)
    '''
    
    def __init__(self,labtime=False):
        
        if labtime:
            
            self.runs = np.genfromtxt('/cds/home/d/dtuthill/LY72_Spooktroscopy/lab_time_runs.csv',delimiter=',',names=True)
            
            self.files = [str(run_set[0].astype(int)) + '-' + str(run_set[1].astype(int)) for run_set in self.runs]
            
            self.runs_input = [np.arange(run_set[0].astype(int),run_set[1].astype(int)+1) for run_set in self.runs]
            
            self.dlys = dly_convert(self.runs_input,import_table=True)
        
        else:
        
            with open('/cds/home/d/dtuthill/LY72_Spooktroscopy/runs_delay.pkl','rb') as f:
                self.runs = pickle.load(f)

            self.files = [delay.replace('.','') for delay in list(self.runs.keys())]

            delays_f_rf, runs_input_rf, delays_f_rt, runs_input_rt, run_dict = get_XAS_runs_LY72_corrected(max_delay_diff=1e-3, compute_delay_ver='v2')
            self.dlys = np.concatenate((delays_f_rt, delays_f_rf))
            self.runs_input = runs_input_rt + runs_input_rf
            reamplified = np.concatenate((np.ones(len(delays_f_rt)), np.zeros(len(delays_f_rf))))
        