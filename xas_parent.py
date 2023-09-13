"""
Class for running XAS algorithm. Call this to initialize arrays
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
import h5py
from tqdm.notebook import tqdm
import sys
import time

import spook

import classes
from analysis_functions import *

sys.path.append('/cds/home/j/jcryan/X400/')
from Time2Energy import T2E

sys.path.append('/cds/home/i/isele/campaign/libraries')
import fzp_ana

class XAS:
    def __init__(self,labtime=False):
        #initialize all subclasses
        
        self.runs = classes.runs(labtime)
            
        self.norms = classes.normalizations()
        
        self.masks = classes.masks()
        
        self.params = classes.params()
        
    def load_data(self,save=False,g_ints_include=False,path='/cds/home/d/dtuthill/LY72_Spooktroscopy/xas/datasets/no_rebin'):
        
        '''
        loads in data as dictionaries keyed by file names
        '''
        #data must first be preprocessed via dataset_creation_v2.py
        
        self.A = {} #spectrometer data
        self.b = {} #bucket electron counts
        self.e_pump = {} #pump energy
        self.e_probe = {} #probe energy
        self.g_ints = {} #fzp spectrometer filter fit values
        self.g_ints_err = {} #fzp spectrometer filter fit errors
        self.path = path
        self.pe_mask = {} #photon energy mask that removes infs, nans
        
        for i,file in enumerate(tqdm(self.runs.files,desc='Loading data')):
            
            #there are four different preprocessing types
            #1. (no_rebin_labtime) this separates same delay configurations that were taken at different labtimes but filters on spectrometer amplitudes
            #2. (labtime_no_filter) 1 but with no filter
            #3. (grouped_no_filter) same delay configurations are grouped into one dataset
            #4. (no_rebin_grouped) 3 but with filters on spectrometer amplitudes

            data_path = path+'/{}.h5'.format(file)
            
            #method 1 and 4
            h5 = h5py.File(data_path, 'r')
            if path[-7:] == 'labtime':
                #method 1 A load in
                self.A[file] = np.array(h5['A'])#*self.norms.yag
            else:
                #method 4 A load in
                self.A[file] = np.array(h5['A'])*self.norms.yag
            #method 1 and 4 load in is the same (note pe_mask, g_ints, g_ints_err are not saved in this preprocessing)
            self.e_pump[file] = np.array(h5['e_pump'])
            self.e_probe[file] = np.array(h5['e_probe'])
            self.b[file] = np.array(h5['b'])

            #method 2
            if path[-17:] == 'labtime_no_filter':
                mask = np.array(h5['pump_mask']).astype(bool) & np.array(h5['probe_mask']).astype(bool)

                self.A[file] = self.A[file][mask]
                self.e_pump[file] = self.e_pump[file][mask]
                self.e_probe[file] = self.e_probe[file][mask]
                self.b[file] = self.b[file][mask]
                self.pe_mask[file] = mask

            #method 3
            elif path[-17:] == 'grouped_no_filter':
                mask = np.array(h5['pump_mask']).astype(bool) & np.array(h5['probe_mask']).astype(bool)

                self.A[file] = self.A[file][mask]
                self.e_pump[file] = self.e_pump[file][mask]
                self.e_probe[file] = self.e_probe[file][mask]
                self.b[file] = self.b[file][mask]
                self.pe_mask[file] = mask

            if g_ints_include:

                self.g_ints[file] = np.array(h5['g_ints'])
                self.g_ints_err[file] = np.array(h5['g_ints_err'])

            h5.close()

        self.fzp_eV = pix2eV(np.linspace(0,1023,self.A[file].shape[-1]))

        self.und_k = {}
        self.bpmx = {}

        #load in beam position monitor values and k's
        #This should be moved to data preprocessing in future updates
        for i,runs in enumerate(tqdm(self.runs.files,desc='loading bpm')):

            und_vals_tot = np.zeros((48,0))
            und_bpmx_tot = np.zeros((52,0))

            for run in self.runs.runs_input[i]:

                gmd_path = '/cds/data/psdm/tmo/tmo%s/scratch/preproc/v7/run%d.h5' %('ly7220', run)
                gmd_file = h5py.File(gmd_path, 'r')

                und_vals = np.zeros((48,np.array(gmd_file['timestamp']).shape[0]))

                #undulator 35 is chicane and hence no k measurement here
                for und in np.arange(26,48):
                    if und == 35:
                        und_vals[35] = np.nan
                        continue

                    #save undulator k's
                    und_var = 'epics_UND_%d_k' %(und)
                    und_vals[und] = np.array(gmd_file[und_var])

                und_bpmx = np.zeros((52,np.array(gmd_file['timestamp']).shape[0]))

                #save undulator bpms
                for und in np.arange(16,52):
                    #there are not bpm monitors at these undulators
                    if und in [17,18,20,22,23,48,49,50]:
                        und_bpmx[und] = np.nan
                        continue

                    und_var = 'epics_bpm_%d_' %(und)
                    und_bpmx[und] = np.array(gmd_file[und_var+'X'])

                und_vals_tot = np.append(und_vals,und_vals_tot,axis=1)
                und_bpmx_tot = np.append(und_bpmx,und_bpmx_tot,axis=1)

                gmd_file.close()

            self.und_k[runs] = und_vals_tot[:,self.pe_mask[runs]]
            self.bpmx[runs] = und_bpmx_tot[:,self.pe_mask[runs]]
        
    def rebin_data(self):
        
        '''
        rebins A data from 1024 bins to user input bins
        '''
        
        for i, file in enumerate(self.runs.files):
            
            self.A[file],transfer_matrix,_,new_bins = rebin(self.A[file],self.params.nbins)
            
        #also need to rebin spectrometer axis in eV
        self.fzp_eV = pix2eV(new_bins)
                
        #also need to rebin regression dillution correction
        self.norms.TtT_norm = transfer_matrix.T @ (self.norms.TtT_norm @ transfer_matrix)
        
    def rebin_yag(self):

        '''
        rebins YAG transmission function
        Note: YAG TF is applied either in preprocessing or data load in and hence should not normally need to be rebinned
        '''
        self.norms.yag,_,_,_ = rebin(self.norms.yag,self.params.nbins)
        
    def mask_data(self,method='fit'):
        
        '''
        Mask the data removing shots where the spectrometer reading was cut by edges
        '''
        
        for i, file in enumerate(self.runs.files):
            
            if method == 'fit':

                self.A[file] = self.A[file][self.masks.edge_masks[file]]
                self.b[file] = self.b[file][self.masks.edge_masks[file]]
                self.e_pump[file] = self.e_pump[file][self.masks.edge_masks[file]]
                self.e_probe[file] = self.e_probe[file][self.masks.edge_masks[file]]
                self.und_k[file] = self.und_k[file][:,self.masks.edge_masks[file]]
                self.bpmx[file] = self.bpmx[file][:,self.masks.edge_masks[file]]
                
            elif method == 'erik':
                
                fzp_zoom, fzp_sum, fzp_t, fzp_com, fzp_filt = fzp_ana.fzp_analyze_v2(self.A[file], thresh=1.6, filt_sum_com=False, hwhm=50)
                
                self.A[file] = self.A[file][fzp_filt]
                self.b[file] = self.b[file][fzp_filt]
                self.e_pump[file] = self.e_pump[file][fzp_filt]
                self.e_probe[file] = self.e_probe[file][fzp_filt]
                self.und_k[file] = self.und_k[file][:,fzp_filt]
                self.bpmx[file] = self.bpmx[file][:,fzp_filt]
            
#     def mask_data_bpmx(self):
        
#         for i, file in enumerate(self.runs.files):
            
#             if file[0] != '1':
#                 first_run = int(file[:2])
#             else:
#                 first_run = int(file[:3])

#             if (first_run >= 116) & (first_run < 137):
#                 pass
#             else:
#                 continue
                
#             mask1 = (self.bpmx[file][self.params.und_filt]<-0.13)
#             mask2 = (self.bpmx[file][self.params.und_filt]>-0.1)
        
#             mask = ~mask1 & ~mask2

#             self.A[file] = self.A[file][mask]
#             self.b[file] = self.b[file][mask]
#             self.e_pump[file] = self.e_pump[file][mask]
#             self.e_probe[file] = self.e_probe[file][mask]
        
    def set_params(self,**kwargs):
        
        '''
        Set new analysis parameters
        '''
        
        for key, value in kwargs.items():
            
            setattr(self.params, key, value)
            
    def plot_A(self,unmasked=False):
        
        '''
        Plots the average spectrometer measurement for each file
        Unmasked input will also plot the measurement without edge mask
        '''
        
        rows = int(np.ceil(len(self.runs.files)/4))
        
        fig,ax = plt.subplots(rows,4,figsize=(20,5*rows),dpi=250)

        ax = ax.ravel()

        for i,file in enumerate(self.runs.files):

            ax[i].plot(self.fzp_eV,self.A[file].mean(0)/self.A[file].mean(0).max(),color='k')
            
            dly_i = i

            if i == 0:
                ax[i].set_title('probe')
            else:
                ax[i].set_title(self.runs.dlys[dly_i])
                
            ax[i].set_title(str(self.runs.dlys[dly_i])+' fs\n'+str(file), size=12)

            ax[i].set_xlabel('Photon Energy (eV)')
            
        line2 = Line2D([0], [0], label='Edge filtered + yag tf', color='k')

        ax[-1].legend(handles=[line2],loc='center',fontsize=16)  
        
        if unmasked:
            
            for i,file in enumerate(self.runs.files):

                data_path = self.path+'/{}.h5'.format(file)

                h5 = h5py.File(data_path, 'r')
                A_unmsk = np.array(h5['A'])
                h5.close()

                ax[i].plot(pix2eV(np.arange(0,1024)),A_unmsk.mean(0)/A_unmsk.mean(0).max(),color='grey',label='Processed Spectrum')
                
            line1 = Line2D([0], [0], label='Original + yag tf', color='grey')
            ax[-1].legend(handles=[line1,line2],loc='center',fontsize=16)  
            
        fig.tight_layout()
            
    def create_pump_bins(self,adaptive=True):
        
        '''
        For each file, separate data into bins according to pump pulse energy
        Adaptive: just set # of bins and upper and lower cutoff energies and the exact bin edges are adaptively set to have the same # of shots per bin
        Not adaptive: The bins are even energy widths but the # of shots is not even per bin
        '''
        
        self.pump_idx = {}
        
        self.pump_sorted = {}
        
        self.pump_b_mean_no_bpmx = {}
        
        if adaptive:
                
            for i,file in enumerate(tqdm(self.runs.files,desc='Calculating bins')):

                pump_b_mean_file = np.zeros(self.params.pump_bins)

                #remove shots first that are above bin extrema
                mask = (self.e_pump[file] >= self.params.pump_min) & (self.e_pump[file] <= self.params.pump_max)

                masked_pump = self.e_pump[file][mask]

                #interpolate array with x=integer indices, y=sorted by value pump energies
                #Evaluate interpolation at integer indices that evenly divide data by shots according to # of bins
                #this evaluation is bin edges
                npt = masked_pump.shape[0]
                self.pump_sorted[file] = np.interp(np.linspace(0.00,npt,self.params.pump_bins+1),
                         np.arange(npt),
                         np.sort(masked_pump))

                #sort data into bins according to bin edges above
                self.pump_idx[file] = np.digitize(self.e_pump[file],self.pump_sorted[file])

                #save mean pump energies of each bin
                for j in np.arange(1,self.params.pump_bins+1):

                    pump_b_mean_file[j-1] = self.e_pump[file][self.pump_idx[file] == j].mean()

                self.pump_b_mean_no_bpmx[file] = pump_b_mean_file
                    
            
        else:
            
            for i,file in enumerate(tqdm(self.runs.files,desc='Calculating pump bins')):
                
                #set even bin edges
                self.pump_sorted[file] = np.linspace(self.params.pump_min,self.params.pump_max,self.params.pump_bins+1)
            
                #sort data into bins
                self.pump_idx[file] = np.digitize(self.e_pump[file],np.linspace(self.params.pump_min,self.params.pump_max,self.params.pump_bins+1))
                
                pump_b_mean_file = np.zeros(self.params.pump_bins)
                
                #calculate mean pump energy of bins
                for j in np.arange(1,self.params.pump_bins+1):
                    
                    pump_b_mean_file[j-1] = self.e_pump[file][self.pump_idx[file] == j].mean()
                
                self.pump_b_mean_no_bpmx[file] = pump_b_mean_file
                
    def create_bpmx_bins(self):
        
        '''
        For each file, separate data into bins according to bpmx
        Adaptive is only option of setting bins
        Adaptive: just set # of bins and upper and lower cutoff bpmx and the exact bin edges are adaptively set to have the same # of shots per bin
        '''
        
        self.bpmx_idx = {}
        
        self.bpmx_sorted = {}
        
        self.bpmx_mean_no_pump = {}                
                
        for i,file in enumerate(tqdm(self.runs.files,desc='Calculating bins')):

            bpmx_mean_file = np.zeros(self.params.bpmx_bins)

            #remove shots first that are above bin extrema
            mask = (self.bpmx[file][self.params.und_filt] >= self.params.bpmx_min) & (self.bpmx[file][self.params.und_filt] <= self.params.bpmx_max)

            masked_bpmx = self.bpmx[file][self.params.und_filt][mask]

            #interpolate array with x=integer indices, y=sorted by value bpmx
            #Evaluate interpolation at integer indices that evenly divide data by shots according to # of bins
            #this evaluation is bin edges
            npt = masked_bpmx.shape[0]
            self.bpmx_sorted[file] = np.interp(np.linspace(0.00,npt,self.params.bpmx_bins+1),
                     np.arange(npt),
                     np.sort(masked_bpmx))

            #sort data into bins according to bin edges above
            self.bpmx_idx[file] = np.digitize(self.bpmx[file][self.params.und_filt],self.bpmx_sorted[file])

            #calculate mean pump energy of bins
            for j in np.arange(1,self.params.bpmx_bins+1):

                bpmx_mean_file[j-1] = self.bpmx[file][self.params.und_filt][self.bpmx_idx[file] == j].mean()

            self.bpmx_mean_no_pump[file] = bpmx_mean_file
            
    def create_bpmx_pump_means(self):
        
        '''
        Calculate average pump energy and bpmx energy for 2d bins (bpmx,pump_energy) 
        '''
        
        self.pump_b_mean = {}
        self.bpmx_b_mean = {}
        
        for i,file in enumerate(tqdm(self.runs.files,desc='Calculating pump bins')):
        
            pump_b_mean_file = np.zeros((self.params.pump_bins,self.params.bpmx_bins))
            bpmx_b_mean_file = np.zeros((self.params.pump_bins,self.params.bpmx_bins))

            for m in np.arange(1,self.params.bpmx_bins+1):

                for j in np.arange(1,self.params.pump_bins+1):
                    
                    #select 2d bin
                    mask = (self.pump_idx[file] == j) & (self.bpmx_idx[file] == m)

                    #take means
                    pump_b_mean_file[j-1,m-1] = self.e_pump[file][mask].mean()
                    bpmx_b_mean_file[j-1,m-1] = self.bpmx[file][self.params.und_filt,mask].mean()

            self.pump_b_mean[file] = pump_b_mean_file
            self.bpmx_b_mean[file] = bpmx_b_mean_file
            
        
#     def lsmooth_scan(self,smooths = np.logspace(-5,5,36)):
        
#         for i,file in enumerate(tqdm(self.runs.files,desc='Calculating X')):

#             fig,ax = plt.subplots(6,6,figsize=(20,20),dpi=30)
#             ax = ax.ravel()
#             fig.suptitle(file)
            
#             for m in np.arange(1,self.params.bpmx_bins+1):
#                 for k in np.arange(1,self.params.pump_bins+1):
                    
#                     mask = (self.pump_idx[file] == k) & (self.bpmx_idx[file] == m)
                    
#                     if (self.pump_idx[file] == k).sum() != 0:
#                         A_split = self.A[file][mask]
#                         b_split = self.b[file][mask]

#                         AtA_norm = (A_split.T @ A_split)/(len(A_split))
#                         StS_norm = AtA_norm - self.norms.TtT_norm # Subtract the contribution from FZP noise
#                         AtB_norm = (A_split.T @ b_split)/(len(A_split))

#                         spokane = spook.SpookPosL1(AtB_norm, StS_norm, "contracted", lsparse = 0, lsmooth=(smooths[0],smooths[0]))

#                         for jdx,j in enumerate(smooths):
#                             spokane.solve(0,(j,j))
#                             X = spokane.getXopt()
#                             ax[jdx].plot(self.fzp_eV,X,color=cm.cool(k/self.params.pump_bins))
#                             ax[jdx].set_title(r'$\lambda _{{sp}}$: {:.1E}, $\lambda _{{sm}}$: {:.1E}'.format(0, j), size=12)
#                             #ax[jdx].plot(fzp_eV,A.mean(0)*0.005,color='black')
#                             ax[jdx].set_ylim([0,0.0005])

#                 fig.tight_layout()
            
    def L_curve_select(self,smooths = np.logspace(-6,6,19),plot=False,x0=-1,x3=6):
        
        '''
        Optimization of hyperparameters for Tikhonov regularization
        Uses the algorithm from DOI 10.1088/2633-1357/abad0d
        See L_curve_search in analysis_functions.py for explanation of algorithm
        '''

        if plot:

            plot_figs = {}
            
            #initial figure for plotting
            for m in range(self.params.bpmx_bins):
                rows = int(np.ceil(len(self.runs.files)/4))
                fig,ax = plt.subplots(rows,4,figsize=(20,rows*5))#,figsize=(20,rows*5),dpi=30)
                # ax = ax.ravel()

                plot_figs[m] = fig

        self.smooth = {}

        for i,file in enumerate(tqdm(self.runs.files,desc='Calculating L-curves')):
            
            #initialize smoothness hyperparameter array
            smoothness = np.zeros((self.params.pump_bins,self.params.bpmx_bins))

            for m in np.arange(1,self.params.bpmx_bins+1):

                for k in np.arange(1,self.params.pump_bins+1):
                    
                    #select files in bpmx/pump bin
                    mask = (self.pump_idx[file] == k) & (self.bpmx_idx[file] == m)
                    
                    #mask A and b
                    A_split = self.A[file][mask]
                    b_split = self.b[file][mask]
                    
                    #precontract matrices for spooktroscopy
                    #regression dilution correction also added
                    AtA_norm = (A_split.T @ A_split)/(len(A_split))
                    StS_norm = AtA_norm - self.norms.TtT_norm # Subtract the contribution from FZP noise
                    AtB_norm = (A_split.T @ b_split)/(len(A_split))
                    
                    #perform hyperparameter optimization
                    l,_,_,L2_new,sm_new = L_curve_search(StS_norm,AtB_norm,b_split.T @ b_split,x0=-1,x3=6,epsilon=0.0001,method='golden')

                    smoothness[k-1,m-1] = l

                    if plot:

                        ax = plot_figs[m-1].axes
                        # ax = ax.ravel()

                        # if m != (self.params.bpmx_bins // 2):
                        #     continue
                        
                        spokane = spook.SpookPosL1(AtB_norm, AtA_norm, "contracted", lsparse = 0, lsmooth=(smooths[0],smooths[0]))
                        
                        #iterate through smooths to create tikhonov regularization curve for comparison
                        for lam in smooths:

                            spokane.solve(0,(lam,lam))
                            X = spokane.getXopt().flatten()                                
                            # res = A_split @ X - b_split
                            # L2 = np.linalg.norm(res,2)**2
                            L2 = spokane.residueL2(b_split.T @ b_split)**2
                            lap = np.square(laplacian1D_S(len(X)) @ X)[1:-1].sum()

                            ax[i].scatter(L2_new(L2),sm_new(lap),color=cm.cool((k-1)/self.params.pump_bins))
                        
                        #solve spooktroscopy at chosen smoothness parameter and mark as *
                        spokane.solve(0,(l,l))
                        X = spokane.getXopt().flatten()                                
                        # res = A_split @ X - b_split
                        # L2 = np.linalg.norm(res,2)**2
                        L2 = spokane.residueL2(b_split.T @ b_split)**2
                        lap = np.square(laplacian1D_S(len(X)) @ X)[1:-1].sum()

                        ax[i].scatter(L2_new(L2),sm_new(lap),color=cm.cool((k-1)/self.params.pump_bins),s=500,marker='*')

                        ax[i].set_title(file)

                self.smooth[file] = smoothness

    #     def L_curve(self,smooths = np.logspace(-6,6,19)):

    #         self.smooth = {}

    #         for i,file in enumerate(tqdm(self.runs.files,desc='Creating curves')):

    #              for k in np.arange(1,self.params.pump_bins+1):

    #                 A_split = self.A[file][self.pump_idx[file] == k]
    #                 b_split = self.b[file][self.pump_idx[file] == k]

    #                 AtA_norm = (A_split.T @ A_split)/(len(A_split))
    #                 StS_norm = AtA_norm - self.norms.TtT_norm # Subtract the contribution from FZP noise
    #                 AtB_norm = (A_split.T @ b_split)/(len(A_split))

    #                 spokane = spook.SpookPosL1(AtB_norm, AtA_norm, "contracted")

    #                 l,iterations,save,_,_ = L_curve_search(A_split,b_split,x0=-2,x3=6,epsilon=0.001)

    #                 rows = int(np.ceil(iterations/4))
    #                 fig,ax = plt.subplots(rows,4,figsize=(20,rows*5),dpi=250)
    #                 fig.suptitle(file)

    #                 ax = ax.ravel()

    #                 for m in range(iterations):

    #                     for n in range(4):

    #                         spokane.solve(0,(save[m,n],save[m,n]))
    #                         X = spokane.getXopt().flatten()                                
    #                         # res = A_split @ X - b_split
    #                         # L2 = np.linalg.norm(res,2)**2
    #                         L2 = spokane.residueL2(b_split.T @ b_split)**2
    #                         smoothness = np.square(laplacian1D_S(len(X)) @ X)[3:-3].sum()

    #                         for mm in range(m+1,iterations):
    #                             ax[mm].scatter(L2,smoothness,facecolors='grey',edgecolors='k',zorder=2)

    #                         ax[m].scatter(L2,smoothness,facecolors='red',edgecolors='k',zorder=3)

    #                         ax[m].set_yscale('log')
    #                         ax[m].set_xscale('log')


    #                 # ax[i*self.params.pump_bins+k-1].set_title(file)

    #                 smooth_mat = np.zeros((len(smooths),4))

    #                 for jdx,j in enumerate(smooths):
    #                     spokane.solve(0,(j,j))
    #                     X = spokane.getXopt().flatten()                                
    #                     # res = A_split @ X - b_split
    #                     # L2 = np.linalg.norm(res,2)**2
    #                     L2 = spokane.residueL2(b_split.T @ b_split)**2
    #                     smoothness = np.square(laplacian1D_S(len(X)) @ X)[3:-3].sum()

    #                     for m in range(iterations):

    #                         ax[m].scatter(L2,smoothness,facecolors='lightgrey',edgecolors='lightgrey',zorder=1)

    #                     smooth_mat[jdx] = [j,L2,smoothness,np.sqrt(L2**2+smoothness**2)]


    #                 self.smooth[file] = smooth_mat

    #                 print(file,k,l)
    #                 # ax[i].set_yscale('log')
    #                 # ax[i].set_xscale('log')

    #                 # ax[(i*self.params.pump_bins)+k-1].scatter(L2,smoothness,color=cm.cool((k-1)/self.params.pump_bins),marker='*',s=500)

#         fig.tight_layout()
                        
    def calc_sigma(self):
        
        '''
        This function calcules that number o ffiles in each pump and bpmx bin
        It is called sigma as 1/N is used as uncertainty for averaging
        For adaptive binning these should all be equal +-1
        '''
        
        self.sigma = np.zeros((len(self.runs.files),self.params.pump_bins,self.params.bpmx_bins))
        
        for i,file in enumerate(tqdm(self.runs.files,desc='Calculating sigma')):
                
            for m in np.arange(1,self.params.bpmx_bins+1):
                
                for j in np.arange(1,self.params.pump_bins+1):
                    
                    mask = (self.pump_idx[file] == j) & (self.bpmx_idx[file] == m)
                    A_split = self.A[file][mask]
                    self.sigma[i,j-1,m-1] = A_split.shape[0]
    
    def alpha_scan(self,alpha_bins=20,alpha_max=8,plot=False,plot_individual=False,create=False,save=False):
        
        '''
        alpha scan adaptively chooses the correct scaling factor to remove counts from 3w of the pump
        '''
        
        if create:
            '''
            First we scan through scaling parameters alpha from 0 to alpha_max in alpha_bins steps
            Calculte the non-resonant region absorption as a function of pump energy
            Nonresonant region is just made up of 3w valence and 2w valence
            2w component should be normalized out during spooktroscopy
            Alpha parameter should account for 3w valence ionization via (alpha*pump_energy = 3w valence)
            '''
            
            #initialize array of absorptions
            X_sum = np.empty((len(self.runs.files),self.params.pump_bins,self.params.bpmx_bins,alpha_bins))   

            #ROI for nonresonant region
            roi_low = np.argmin(np.abs(self.fzp_eV-511))
            roi_high = np.argmin(np.abs(self.fzp_eV-520))

            for i,file in enumerate(tqdm(self.runs.files,desc='Scanning alpha')):
                
                # if i < 4:
                #     dly_i = i-1
                # else:
                #     dly_i = i
                
                dly_i = i
                
                # if plot_individual:
                #     fig,ax = plt.subplots(int(np.ceil(alpha_bins/5)),5,figsize=(20,20),dpi=100)
                #     ax = ax.ravel()
                
                for m in np.arange(1,self.params.bpmx_bins+1):

                    for j in np.arange(1,self.params.pump_bins+1):
                        
                        #bin on bpmx/pump
                        mask = (self.pump_idx[file] == j) & (self.bpmx_idx[file] == m)
                        
                        if mask.sum() != 0:
                            A_split = self.A[file][mask]
                            
                            #precontract for spooktroscopy
                            #remove regression dilution
                            AtA_norm = (A_split.T @ A_split)/(len(A_split))
                            StS_norm = AtA_norm - self.norms.TtT_norm

                            for kdx,k in enumerate(np.linspace(0,alpha_max,alpha_bins)):
                                b_split = self.b[file][mask] - self.e_pump[file][mask]*k
                                
                                AtB_norm = (A_split.T @ b_split)/(len(A_split))
                                spokane = spook.SpookPosL1(AtB_norm, StS_norm, "contracted", lsparse = 0,lsmooth = (self.smooth[file][j-1,m-1],self.smooth[file][j-1,m-1]))
                                
                                #solve for absorption
                                spokane.solve(0,(self.smooth[file][j-1,m-1],self.smooth[file][j-1,m-1]))
                                X = spokane.getXopt()
                                X_sum[i,j-1,m-1,kdx] = X[roi_low:roi_high+1].sum()

                                if plot_individual:
                                    
                                    pass
                                    
                                    ##needs to be updated for bpmx_binx

                                    # ax[kdx].plot(self.fzp_eV,X,color=cm.cool(j/self.params.pump_bins))
                                    # ax[kdx].set_xlabel('Photon Energy (eV)')
                                    # ax[kdx].set_title('alpha='+str(k))
                                    # fig.suptitle(file+'\n'+str(self.runs.dlys[dly_i]))
                                    # ax[kdx].set_ylim([0,0.0003])

                        else:
                            X_sum[i,j-1,m-1,kdx] = 0

            '''
            Fit absorption vs. pump energy for each alpha parameter
            '''
            
            #initialize slope array
            lines = np.zeros((len(self.runs.files),self.params.bpmx_bins,alpha_bins))
            self.b_offset = {}

            if plot:

                plot_figs = {}

                for m in range(self.params.bpmx_bins):
                    rows = int(np.ceil(len(self.runs.files)/4))
                    fig,ax = plt.subplots(rows,4,figsize=(20,rows*5))

                    plot_figs[m] = fig

            for i,file in enumerate(self.runs.files):
                
                for m in np.arange(1,self.params.bpmx_bins+1):
                    
                    if plot:
                        
                        ax = plot_figs[m-1].axes

                        dly_i = i
                        ax[i].set_title(self.runs.dlys[dly_i])
                                        
                    for k in range(alpha_bins):
                        
                        #only fit slopes to regions where X != 0 and hence at bottom and not linear (spooktroscopy imposes nonnegativity)
                        max_idx = np.where(X_sum[i,:,m-1,k] > 1e-5)[0][-1]+1
                        if max_idx == 1:
                            continue
                        
                        #fit
                        popt_line,_ = curve_fit(line,self.pump_b_mean[file][:max_idx,m-1],X_sum[i,:,m-1,k][:max_idx],sigma=np.reciprocal(self.sigma[i][:max_idx,m-1]))

                        if plot:                                                                                    
                            ax[i].plot(self.pump_b_mean[file][:,m-1],X_sum[i,:,m-1,k],color=cm.winter(k/(alpha_bins-1)))
                            ax[i].plot(self.pump_b_mean[file][:,m-1],line(self.pump_b_mean[file][:,m-1],*popt_line),
                                       alpha=0.5,color=cm.winter(k/(alpha_bins-1)),linestyle='--')
                            ax[i].set_xlabel('Pump energy (mJ)')
                            ax[i].set_ylabel('511-520 eV absorbance')

                        lines[i,m-1,k] = popt_line[0]

            '''
            Fit slopes of absorption vs. pump energy as a function of alpha parameter
            Alpha parameter that gives slope of 0 (i.e. more pump energy does not give more or less valence ionization) is correct one
            '''

            if plot:

                plot_figs = {}

                for m in range(self.params.bpmx_bins):
                    rows = int(np.ceil(len(self.runs.files)/4))
                    fig,ax = plt.subplots(rows,4,figsize=(20,rows*5))

                    plot_figs[m] = fig

            for idx,i in enumerate(self.runs.files):
                
                b_offset = np.zeros(self.params.bpmx_bins)
                
                for m in np.arange(1,self.params.bpmx_bins+1):
                    
                    #again only using linearly changing region
                    max_idx = np.where(lines[idx,m-1] != 0)[0][-1]+1
                    popt_line2,pcov = curve_fit(line,np.linspace(0,alpha_max,alpha_bins)[:max_idx],lines[idx,m-1][:max_idx])
                    print(i,-popt_line2[1]/popt_line2[0],np.diag(pcov))
                    
                    #we want x such that y=0 for mx+b=y
                    b_offset[m-1] = -popt_line2[1]/popt_line2[0]

                    if plot:
                        
                        ax = plot_figs[m-1].axes
                        
                        ax[idx].plot(np.linspace(0,alpha_max,alpha_bins),lines[idx,m-1],color='black')
                        ax[idx].plot(np.linspace(0,alpha_max,alpha_bins),line(np.linspace(0,alpha_max,alpha_bins),*popt_line2),color='red',linestyle='--')

                        dly_i = idx

                        ax[idx].set_title(self.runs.dlys[dly_i])
                        ax[idx].set_xlabel('alpha')
                        ax[idx].set_ylabel(r'$\Delta$ 511-520 eV absorbance')


                self.b_offset[i] = b_offset
            
            #save for reuse later with recalculating (recalculating should only take ~10s)
            if save:
                
                hf_save = h5py.File('/cds/home/d/dtuthill/LY72_Spooktroscopy/xas/b_offset.h5','w')
                hf_save.create_dataset('b_offset', data=list(b_offset.values()))
                hf_save.create_dataset('delays', data=list(b_offset.keys()))
                hf_save.close()

                print('b_offset saved succesfully')
        
        #if not creating than load saved data
        else:
            
            b_offset_path = '/cds/home/d/dtuthill/LY72_Spooktroscopy/xas/b_offset.h5'
            b_offset_file = h5py.File(b_offset_path,'r')
            b_offset_keys = [delay.decode('utf-8') for delay in b_offset_file['delays']]

            self.b_offset = dict(zip(b_offset_keys,b_offset_file['b_offset']))
            b_offset_file.close()
    
    def calculate_X(self,plot=False):
        
        '''
        Calculate absorption using all previously selected parameters and corrections
        '''

        if plot:

            plot_figs = {}

            for m in range(self.params.bpmx_bins):
                rows = int(np.ceil(len(self.runs.files)/4))
                fig,ax = plt.subplots(rows,4,figsize=(20,rows*5))

                plot_figs[m] = fig

        #resonant region ROI
        low_idx = np.argmin(np.abs(self.fzp_eV-524))
        high_idx = np.argmin(np.abs(self.fzp_eV-530))
        
        #nonresonant region ROI
        low_idx_off = np.argmin(np.abs(self.fzp_eV-515))
        high_idx_off = np.argmin(np.abs(self.fzp_eV-520))
        
        #in between region ROI
        low_idx_mid = np.argmin(np.abs(self.fzp_eV-520))
        high_idx_mid = np.argmin(np.abs(self.fzp_eV-525))
        
        #initialize absorption of each region
        self.X_sum = np.zeros((len(self.runs.files),self.params.pump_bins,self.params.bpmx_bins))
        self.X_sum_off = np.zeros_like(self.X_sum)
        self.X_sum_mid = np.zeros_like(self.X_sum)

        for i,file in enumerate(tqdm(self.runs.files,desc='Calculating X')):

            # if i < 4:
            #     dly_i = i-1
            # else:
            #     dly_i = i
            
            dly_i = i
            
            for m in np.arange(1,self.params.bpmx_bins+1):
                
                ax = plot_figs[m-1].axes
            
                for k in np.arange(1,self.params.pump_bins+1):
                    
                    #mask on bpmx and pump
                    mask = (self.pump_idx[file] == k) & (self.bpmx_idx[file] == m)
                    if mask.sum() != 0:
                        
                        A_split = self.A[file][mask]
                        
                        #remove 3w component
                        b_split = self.b[file][mask] - self.e_pump[file][mask]*self.b_offset[file][m-1]

                        #precontract
                        AtA_norm = (A_split.T @ A_split)/(len(A_split))
                        StS_norm = AtA_norm - self.norms.TtT_norm # Subtract the contribution from FZP noise
                        AtB_norm = (A_split.T @ b_split)/(len(A_split))
                        
                        #calculate absorption
                        spokane = spook.SpookPosL1(AtB_norm, StS_norm, "contracted", lsparse = 0,lsmooth = (self.smooth[file][k-1,m-1],self.smooth[file][k-1,m-1]))
                        spokane.solve(0,(self.smooth[file][k-1,m-1],self.smooth[file][k-1,m-1]))
                        X = spokane.getXopt().flatten()
                        
                        #sum absorption in ROIs
                        self.X_sum[i,k-1,m-1] = X[low_idx:high_idx].sum()
                        self.X_sum_off[i,k-1,m-1] = X[low_idx_off:high_idx_off].sum()
                        self.X_sum_mid[i,k-1,m-1] = X[low_idx_mid:high_idx_mid].sum()

                        if plot:
                            ax[i].plot(self.fzp_eV,X,color=cm.cool(k/self.params.pump_bins))
                            ax[i].set_title(str(self.runs.dlys[dly_i])+' fs\n'+str(file), size=12)
                            ax[i].set_xlabel('Photon Energy (eV)')
                            ax[i].set_ylabel('Absorption')
                            ax[i].set_ylim([0,0.0006])

                if plot:
                    plot_figs[m-1].tight_layout()
                    
    def fit_X(self,plot=False):
        
        '''
        Using the previously calculated absorptions vs. bpmx and pump energy we calculate absorption vs. pump energy for each bpmx bin
        We are in the end interested in absorption/pump energy slope vs. delay
        '''
        
        if plot:

            fig_all,ax_all = plt.subplots(figsize=(10,3),dpi=200)
            
            plot_figs_X = {}
            plot_figs_off = {}
            plot_figs_on = {}
            plot_figs_mid = {}
            plot_figs_on_b = {}
            
            #loop through bpmx bins
            for m in range(self.params.bpmx_bins):
                rows = int(np.ceil(len(self.runs.files)/4))
                fig,ax = plt.subplots(rows,4,figsize=(20,rows*5))
                
                #initialize plots
                #plot_figs_X are absorption vs. pump energy with fit for every delay
                #plot_figs_on is resonant absorption slope vs delay
                #plots_figs_mid is middle ROI absorption slope vs delay
                #plots_figs_off is off resonant absorption slope vs delay
                #see calculate_X for region definition

                plot_figs_X[m] = fig
                
                fig, ax = plt.subplots()
                plot_figs_on[m] = fig
                
                ax.set_xlabel('Delay (fs)')
                ax.set_ylabel(r'$\Delta$ absorption / $\Delta$ pump energy\n Resonance')            
                
                fig, ax = plt.subplots()
                plot_figs_on_b[m] = fig
                
                ax.set_xlabel('Delay (fs)')
                ax.set_ylabel(r'$\Delta$ absorption / $\Delta$ pump energy\n offset')
                
                fig, ax = plt.subplots()
                plot_figs_mid[m] = fig
                
                ax.set_xlabel('Delay (fs)')
                ax.set_ylabel(r'$\Delta$ absorption / $\Delta$ pump energy\n Mid')
                
                fig, ax = plt.subplots()
                plot_figs_off[m] = fig
                
                ax.set_xlabel('Delay (fs)')
                ax.set_ylabel(r'$\Delta$ absorption / $\Delta$ pump energy\n Off')

#             fig5, ax5 = plt.subplots(dpi=150)
#             fig2, ax2 = plt.subplots(dpi=150)
#             fig3, ax3 = plt.subplots(dpi=150)
#             fig4, ax4 = plt.subplots(dpi=150)
            
#             rows = int(np.ceil(len(self.runs.files)/4))
#             fig1,ax1 = plt.subplots(rows,4,figsize=(20,rows*5),dpi=250)
#             ax1= ax1.ravel()
            
#             # ax1.set_xlabel('Pump Energy (mJ)')
#             # ax1.set_ylabel('Absorption 525-530eV')
#             ax2.set_xlabel('Delay (fs)')
#             ax2.set_ylabel(r'$\Delta$ absorption / $\Delta$ pump energy')# (525-530eV)')            
#             ax3.set_xlabel('Pump Energy (mJ)')
#             ax3.set_ylabel('Absorption 525-530eV')
#             ax4.set_xlabel('Pump Energy (mJ)')
#             ax4.set_ylabel('Absorption 520-525eV')
            
#             triangle = Line2D([], [], color='k', marker='^', linestyle='None', alpha=0.5, markerfacecolor='white',
#                           markersize=10, label='520-525 eV')
            
#             closed_circle = Line2D([], [], color='k', marker='o', linestyle='None', markerfacecolor='k',
#                           markersize=10, label='525-530 eV')
            
#             open_circle = Line2D([], [], color='k', marker='o', alpha = 0.5, linestyle='None', markerfacecolor='white',
#                           markersize=10, label='511-520 eV')
            
            #assign colors for different labtime scans
            grey = Line2D([], [], color='grey', marker='o', alpha = 0.6, linestyle='None',
                          markersize=10, label='First runs (bad)')
        
            red = Line2D([], [], color='red', marker='o', alpha = 0.6, linestyle='None',
                          markersize=10, label='Scan 1')
            
            blue = Line2D([], [], color='blue', marker='o', alpha = 0.6, linestyle='None',
                          markersize=10, label='Scan 2')
            
            green = Line2D([], [], color='green', marker='o', alpha = 0.6, linestyle='None',
                          markersize=10, label='Long delays')
            
            purple = Line2D([], [], color='purple', marker='o', alpha = 0.6, linestyle='None',
                          markersize=10, label='Reamplified short delays')
            
            orange = Line2D([], [], color='orange', marker='o', alpha = 0.6, linestyle='None',
                          markersize=10, label='Nonreamplified short delays')
            
            symbols = np.array(['v','o','^'])
            
            # ax2.legend(handles=[open_circle,triangle,closed_circle])
            
                        # ax2.legend(handles=[grey,red,blue,green,purple,orange],bbox_to_anchor=(1.05,1))
        
        #initialize absorption/pump energy vs delay arrays
        self.m = np.zeros((len(self.runs.files),self.params.bpmx_bins,2))

        for m in range(self.params.bpmx_bins):
            
            ax_X = plot_figs_X[m].axes
            ax_off = plot_figs_off[m].axes[0]
            ax_on = plot_figs_on[m].axes[0]
            ax_mid = plot_figs_mid[m].axes[0]
            ax_on_b = plot_figs_on_b[m].axes[0]
            
            slope_avg = {}

            slope = {}
            slope_e = {}
            s = {}

            for dly in np.unique(self.runs.dlys):
                slope[dly] = []
                slope_e[dly] = []
                s[dly] = []
        
            for i,file in enumerate(self.runs.files):
                
                #calculate absorption/pump energy slopes for ROIs for each delay
                popt_line,pcov = curve_fit(line,self.pump_b_mean[file][:,m],self.X_sum[i,:,m],sigma=np.reciprocal(self.sigma[i,:,m]))
                
                popt_line_mid,pcov_mid = curve_fit(line,self.pump_b_mean[file][:,m],self.X_sum_mid[i,:,m],sigma=np.reciprocal(self.sigma[i,:,m]))
                
                popt_line_off,pcov_off = curve_fit(line,self.pump_b_mean[file][:,m],self.X_sum_off[i,:,m],sigma=np.reciprocal(self.sigma[i,:,m]))

                self.m[i,m,0] = popt_line[0]
                self.m[i,m,1] = np.sqrt(np.diag(pcov)[0])

                if plot:

                    dly_i = i

                    ax_X[i].plot(self.pump_b_mean[file][:,m],line(self.pump_b_mean[file][:,m],*popt_line),
                             alpha=0.5,color='k',linestyle='--')
                    ax_X[i].plot(self.pump_b_mean[file][:,m],self.X_sum[i,:,m],color='k')

                    ax_X[i].set_title(file+'\n'+str(self.runs.dlys[dly_i]))
                    
                    middle = popt_line[0]*0.05+popt_line[1]
                    
                    ax_X[i].set_ylim([middle-0.0005,middle+0.0005])

                    #need to look at which labtime grouping this is
                    #if it is short or long then we want to weighted average the same delays
                    if self.runs.runs_input[i][0] < 91:
                        color= 'grey'
                        continue
                    elif (self.runs.runs_input[i][0] >= 91) & (self.runs.runs_input[i][0] < 116):
                        color='red'
                        
                        ax_all.errorbar(self.runs.dlys[dly_i]+0.1*m,popt_line[0],yerr=np.sqrt(np.diag(pcov)[0]),color=color,alpha=0.5,fmt=symbols[m])
                        
                    elif (self.runs.runs_input[i][0] >= 116) & (self.runs.runs_input[i][0] < 137):
                        color='blue'
                        
                        ax_all.errorbar(self.runs.dlys[dly_i]+0.1*m,popt_line[0],yerr=np.sqrt(np.diag(pcov)[0]),color=color,alpha=(1/self.params.bpmx_bins)*(m+1),fmt=symbols[m])
                        
                    elif (self.runs.runs_input[i][0] >= 137) & (self.runs.runs_input[i][0] < 151):
                        color='green'
                        
                        dly = self.runs.dlys[i]

                        slope[dly].append(self.m[i,m,0])
                        slope_e[dly].append(self.m[i,m,1])
                        s[dly].append(self.sigma[i,:,m].sum())
                        
                    elif (self.runs.runs_input[i][0] >= 161) & (self.runs.runs_input[i][0] < 181):
                        color='purple'
                        dly = self.runs.dlys[i]

                        slope[dly].append(self.m[i,m,0])
                        slope_e[dly].append(self.m[i,m,1])
                        s[dly].append(self.sigma[i,:,m].sum())
                        
                    elif (self.runs.runs_input[i][0] >= 181) & (self.runs.runs_input[i][0] < 185):
                        color='orange'
                        continue

                    # ax2.errorbar(self.runs.dlys[dly_i],popt_line[0],yerr=np.sqrt(np.diag(pcov)[0]),color=cm.cool(i/len(self.runs.files)),fmt='o')
                    ax_on.errorbar(self.runs.dlys[dly_i],popt_line[0],yerr=np.sqrt(np.diag(pcov)[0]),color=color,alpha=(1/self.params.bpmx_bins)*(m+1),fmt='o')
                    ax_on_b.errorbar(self.runs.dlys[dly_i],popt_line[1],yerr=np.sqrt(np.diag(pcov)[1]),color=color,alpha=0.6,fmt='o')
                    
                    ax_mid.errorbar(self.runs.dlys[dly_i],popt_line_mid[0],yerr=np.sqrt(np.diag(pcov_mid)[0]),color=color,alpha=0.6,fmt='o')
                    ax_off.errorbar(self.runs.dlys[dly_i],popt_line_off[0],yerr=np.sqrt(np.diag(pcov_off)[0]),color=color,alpha=0.6,fmt='o')

            for dly in slope:

                if not slope[dly]:
                    continue
                    
                if dly < 1.3:
                    color='purple'
                elif dly > 10:
                    color='green'
                    

                slope_avg[dly] = [np.average(slope[dly],weights=s[dly]),np.sqrt(np.square(np.divide(slope_e[dly],slope[dly])).sum())*np.average(slope[dly],weights=s[dly])]
                
                ax_all.errorbar(dly+0.1*m,slope_avg[dly][0],yerr=slope_avg[dly][1],color=color,alpha=(1/self.params.bpmx_bins)*(m+1),fmt=symbols[m])

            # fig1.tight_layout()

            # ax2.set_ylim([-0.01,0.01])

            # lw88 = np.load('/cds/home/d/dtuthill/tmolw8819/results/erik/spook/sub_3w/sig2.npz')

            # ax2.errorbar(lw88['dlys_updated'],lw88['sig2']*0.6,yerr=lw88['sig2_err']*0.6,color='k',fmt='o',alpha=0.6)
            
#         for i,file in enumerate(self.runs.files):

#             # if i == 0:
#             #     continue
                
#             if (file == '325') or (file == '375'):
#                 continue
            
#             popt_line,pcov = curve_fit(line,self.pump_b_mean[file],self.X_sum_off[i],sigma=np.reciprocal(self.sigma[i]))
            
#             if plot:
#                 ax3.plot(self.pump_b_mean[file],line(self.pump_b_mean[file],*popt_line),
#                          alpha=0.5,color=cm.cool(i/len(self.runs.files)),linestyle='--')
#                 ax3.plot(self.pump_b_mean[file],self.X_sum_off[i],color=cm.cool(i/len(self.runs.files)))

#                 # if i < 4:
#                 #     dly_i = i-1
#                 # else:
#                 #     dly_i = i
                
#                 dly_i = i

#                 # ax2.errorbar(self.runs.dlys[dly_i],popt_line[0],yerr=np.sqrt(np.diag(pcov)[0]),color=cm.cool(i/len(self.runs.files)),fmt='o',alpha=0.5,markerfacecolor='white')
        
        
#             # if i == 0:
#             #     continue
            
#             popt_line,pcov = curve_fit(line,self.pump_b_mean[file],self.X_sum_mid[i],sigma=np.reciprocal(self.sigma[i]))
            
#             if plot:
#                 ax4.plot(self.pump_b_mean[file],line(self.pump_b_mean[file],*popt_line),
#                          alpha=0.5,color=cm.cool(i/len(self.runs.files)),linestyle='--')
#                 ax4.plot(self.pump_b_mean[file],self.X_sum_mid[i],color=cm.cool(i/len(self.runs.files)))

#                 # if i < 4:
#                 #     dly_i = i-1
#                 # else:
#                 #     dly_i = i
                
#                 dly_i = i

#                 # ax2.errorbar(self.runs.dlys[dly_i],popt_line[0],yerr=np.sqrt(np.diag(pcov)[0]),color=cm.cool(i/len(self.runs.files)),fmt='^',alpha=0.75,markerfacecolor='grey')
              
            
#     def labtime_diagnostics(self):
        
#         fig1, ax1 = plt.subplots(dpi=150)
#         fig2, ax2 = plt.subplots(dpi=150)
#         fig3, ax3 = plt.subplots(dpi=150)
#         fig4, ax4 = plt.subplots(dpi=150)
#         fig5, ax5 = plt.subplots(dpi=150)
#         fig6, ax6 = plt.subplots(dpi=150)
#         fig7, ax7 = plt.subplots(dpi=150)
#         fig8, ax8 = plt.subplots(dpi=150)
#         fig9, ax9 = plt.subplots(dpi=150)
#         fig10, ax10 = plt.subplots(dpi=150)
#         fig11, ax11 = plt.subplots(dpi=150)
#         fig12, ax12 = plt.subplots(dpi=150)

#         ax1.set_xlabel('Delay (fs)')
#         ax1.set_ylabel(r'$\Delta$ absorption / $\Delta$ pump energy')# (525-530eV)')            

#         ax2.set_xlabel('Delay (fs)')
#         ax2.set_ylabel('Average pump energy [mJ]')# (525-530eV)')            

#         ax3.set_xlabel('Delay (fs)')
#         ax3.set_ylabel('Average probe energy [mJ]')# (525-530eV)')    
        
#         ax3.set_ylim([-0.0050,0.017])

#         ax4.set_xlabel('Delay (fs)')
#         ax4.set_ylabel('Average counts')# (525-530eV)')            

#         ax5.set_xlabel('Delay (fs)')
#         ax5.set_ylabel('FZP COM')# (525-530eV)')    
        
#         ax6.set_xlabel('Delay (fs)')
#         ax6.set_ylabel('Shots')# (525-530eV)')   
        
#         ax7.set_xlabel('Delay (fs)')
#         ax7.set_ylabel(r'Shots $\times$ probe energy')# (525-530eV)')   

#         grey = Line2D([], [], color='grey', marker='o', alpha = 0.6, linestyle='None',
#                       markersize=10, label='First runs (bad)')

#         red = Line2D([], [], color='red', marker='o', alpha = 0.6, linestyle='None',
#                       markersize=10, label='Scan 1')

#         blue = Line2D([], [], color='blue', marker='o', alpha = 0.6, linestyle='None',
#                       markersize=10, label='Scan 2')

#         green = Line2D([], [], color='green', marker='o', alpha = 0.6, linestyle='None',
#                       markersize=10, label='Long delays')

#         purple = Line2D([], [], color='purple', marker='o', alpha = 0.6, linestyle='None',
#                       markersize=10, label='Reamplified short delays')

#         orange = Line2D([], [], color='orange', marker='o', alpha = 0.6, linestyle='None',
#                       markersize=10, label='Nonreamplified short delays')
                    
#         ax1.legend(handles=[grey,red,blue,green,purple,orange],bbox_to_anchor=(1, 0.5))
            
#         for i,file in enumerate(self.runs.files):
            
#             popt_line,pcov = curve_fit(line,self.pump_b_mean[file],self.X_sum[i],sigma=np.reciprocal(self.sigma[i]))
#             popt_line_off,pcov_off = curve_fit(line,self.pump_b_mean[file],self.X_sum_off[i],sigma=np.reciprocal(self.sigma[i]))
#             popt_line_mid,pcov_mid = curve_fit(line,self.pump_b_mean[file],self.X_sum_mid[i],sigma=np.reciprocal(self.sigma[i]))
                
#             dly_i = i

#             if self.runs.runs_input[i][0] < 87:
#                 color= 'grey'
#             elif (self.runs.runs_input[i][0] >= 87) & (self.runs.runs_input[i][0] < 116):
#                 color='red'
#             elif (self.runs.runs_input[i][0] >= 116) & (self.runs.runs_input[i][0] < 137):
#                 color='blue'
#             elif (self.runs.runs_input[i][0] >= 137) & (self.runs.runs_input[i][0] < 151):
#                 color='green'
#             elif (self.runs.runs_input[i][0] >= 161) & (self.runs.runs_input[i][0] < 181):
#                 color='purple'
#             elif (self.runs.runs_input[i][0] >= 181) & (self.runs.runs_input[i][0] < 185):
#                 color='orange'

#             ax1.errorbar(self.runs.dlys[dly_i],popt_line[0],yerr=np.sqrt(np.diag(pcov)[0]),color=color,alpha=0.6,fmt='o')
#             ax8.errorbar(self.runs.dlys[dly_i],popt_line[0],yerr=np.sqrt(np.diag(pcov)[0]),color=color,alpha=0.6,fmt='o')
#             ax9.errorbar(self.runs.dlys[dly_i],popt_line_off[0],yerr=np.sqrt(np.diag(pcov_off)[0]),color=color,fmt='o',alpha=0.6,markerfacecolor='white')
#             ax10.errorbar(self.runs.dlys[dly_i],popt_line_mid[0],yerr=np.sqrt(np.diag(pcov_mid)[0]),color=color,fmt='^',alpha=0.6,markerfacecolor='grey')
            
#             ax2.errorbar(self.runs.dlys[dly_i],self.e_pump[file].mean(),yerr=self.e_pump[file].std(),color=color,alpha=0.6,fmt='o')
#             ax3.errorbar(self.runs.dlys[dly_i],self.e_probe[file].mean(),yerr=self.e_probe[file].std(),color=color,alpha=0.6,fmt='o')
#             ax4.errorbar(self.runs.dlys[dly_i],self.b[file].mean(),yerr=self.b[file].std(),color=color,alpha=0.6,fmt='o')
#             ax5.scatter(self.runs.dlys[dly_i],np.multiply(self.A[file],self.fzp_eV).sum()/self.A[file].sum(),color=color,alpha=0.6)
#             ax6.scatter(self.runs.dlys[dly_i],self.A[file].shape[0],color=color,alpha=0.6)
#             ax7.scatter(self.runs.dlys[dly_i],self.A[file].shape[0]*self.e_probe[file].mean(),color=color,alpha=0.6)
            
#             ax11.errorbar(self.runs.dlys[dly_i],np.average(self.X_sum_off[i],weights=self.sigma[i]),color=color,fmt='o',alpha=0.6,markerfacecolor='white')
            
# #             if file[2] == '-':
# #                 time = float(file[:2])
# #             else:
# #                 time = float(file[:3])
                
# #             ax12.errorbar(time,np.average(self.X_sum_off[i],weights=self.sigma[i]),color=color,fmt='o',alpha=0.6,markerfacecolor='white')
            
#             ax8.set_ylim([-0.02,0.03])
#             ax9.set_ylim([-0.01,0.01])
#             ax10.set_ylim([-0.01,0.01])
#             ax1.set_ylim([-0.003,0.007])
#             ax11.set_xlabel('Delay')
#             ax12.set_xlabel('Run')
#         ax9.set_xlabel('Delay [fs]')
#         ax10.set_xlabel('Delay [fs]')
        
#         ax9.set_ylabel(r'$\Delta$ absorption (515-520)')
#         ax10.set_ylabel(r'$\Delta$ absorption (520-525)')
        
#         ax9.set_ylim([-0.0025,0.0025])
#         ax10.set_ylim([-0.0025,0.003])
                