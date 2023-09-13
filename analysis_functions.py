import numpy as np
import warnings
import sys
import scipy.sparse as sps
import spook

'''
Base functions needed for x-ray absorption analysis code
'''

def pix2eV(x):
    '''
    Converts FZP spectrometer pixel in nm to eV 
    '''
    b = 1/29.527
    a = 504.07
    return b*x+a

def percentOverlap_ss(minA, maxA, minB, maxB):
    '''
    Determines % overlap of region A with region B
    Both regions are given by scalars
    '''
    
    return (min(maxB, maxA)-max(minB, minA))/(maxA-minA)

def rebin(fzp,binN):
    '''
    Rebins array along axis -1 to binN elements
    binN does not need to be an integer of fzp.shape[-1]
    Computes rebin via percent overlap between old bins and new bins
    '''
    
    #determine old and new bin arrays
    old_bins = np.linspace(0,1023,fzp.shape[-1])
    new_bins = np.linspace(0,1023,binN)
    
    old_end = len(old_bins)-1
    
    #transfer matrix will satisfy fzp.old_bins @ transfer_matrix = fzp.new_bins
    transfer_matrix = np.zeros((fzp.shape[-1],binN))
    
    #go elementwise of new bins and determine overlap with old bins
    for idx in range(len(new_bins)-1):
        
        start = new_bins[idx]
        stop = new_bins[idx+1]
        
        # identify beginning of last old fzp bin that overlaps this new energy bin
        while old_end >= 0 and old_bins[old_end] > stop:
            old_end -= 1
        if old_end < len(old_bins)-1:
            old_end += 1
        
        old_start = old_end
        
        # identify beginning of first old fzp bin that overlaps this new energy bin
        while old_start >= 0 and old_bins[old_start] > start:
            old_start -= 1
        if old_start < 0:
            old_start = 0
        
        # print(old_start,old_end)
        for j in range(old_start, old_end):
            # print(idx,j,old_bins[old_start],old_bins[old_start+1],start,stop)
            transfer_matrix[j,idx] = percentOverlap_ss(old_bins[j],old_bins[j+1],start,stop)
            
        old_end = len(old_bins)-1
            
    return fzp @ transfer_matrix, transfer_matrix, old_bins, new_bins

def gauss(x,a,b,c,d):
    '''
    Gaussian function for curve fitting
    Warnings ignored to avoid spamming of terminal for inf, nan values
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return a*np.exp(-0.5*np.square(np.divide(x-b,c)))+d               
        
def line(x,m,b):
    '''
    Line for linear regrssion fitting
    '''
    return m*x+b

def xgmd_gmd_cross(xgmd,gmd,Eph):
    '''
    Written by Greg McCracken
    Takes pulse energy readings from xgmd and gmd and converts to photon energy
    xgmd has N2, gmd has Kr in it
    Written for w,2w photon energy scheme
    '''
    
    #Krypton photon energy and absorption cross section
    Eph_Kr = np.array([243.595, 247.535, 251.538, 255.607, 259.741,263.942, 268.211, 272.549, 276.957, 501.459, 509.57, 517.812, 526.187, 534.698, 543.346, 552.134])
    cross_Kr = np.array([4.944, 4.905, 4.866, 4.828, 4.766, 4.674, 4.583, 4.494, 4.407, 1.848, 1.794, 1.742, 1.692, 1.64, 1.585, 1.532])
    
    #N2 photon energy and absorption cross section
    Eph_N2 = np.array([240.0, 244.0, 248.0, 252.0, 256.0, 260.0, 264.0, 268.0, 272.0, 276.0, 500.0, 504.0, 508.0, 512.0, 516.0, 520.0, 524.0, 528.0, 532.0, 536.0, 540.0, 544.0, 548.0, 552.0, 556.0])
    cross_N2 = np.array([0.255, 0.244, 0.234, 0.225, 0.216, 0.207, 0.199, 0.191, 0.184, 0.177, 0.904, 0.887, 0.871, 0.856, 0.84, 0.825, 0.811, 0.797, 0.783, 0.769, 0.756, 0.744, 0.731, 0.719, 0.707])
    
    #interpolate cross sections
    Kr_w = np.interp(0.5*Eph,Eph_Kr,cross_Kr)
    Kr_2w = np.interp(Eph,Eph_Kr,cross_Kr)
    N2_w = np.interp(0.5*Eph,Eph_N2,cross_N2)
    N2_2w = np.interp(Eph,Eph_N2,cross_N2)
    
    #ratio of cross sections
    XRat = N2_2w/N2_w
    GRat = Kr_2w/Kr_w
    
    #Transmission of gmd and xgmd
    Fx = 1/0.038 
    Fg = 1

    S = (XRat-GRat)**(-1)
    e_probe = S*(Fx*xgmd - Fg*gmd)
    e_pump = Fg*gmd - GRat*e_probe
    
    return e_pump,e_probe

def hist_laxis(data, n_bins, range_limits):
    '''
    Performs a histogram along each row of data
    '''
    
    # Setup bins and determine the bin location for each element for the bins
    R = range_limits
    bins = np.linspace(R[0],R[1],n_bins+1)
    #reduce N-dimensional array to 2-D
    N = data.shape[-1]
    idx = np.searchsorted(bins, data,'right')-1

    # Some elements would be off limits, so get a mask for those
    bad_mask = (idx==-1) | (idx==n_bins)

    # We need to use bincount to get bin based counts. To have unique IDs for
    # each row and not get confused by the ones from other rows, we need to 
    # offset each row by a scale (using row length for this).
    scaled_idx = n_bins*np.arange(data.shape[0])[:,None] + idx

    # Set the bad ones to be last possible index+1 : n_bins*data2D.shape[0]
    limit = n_bins*data.shape[0]
    scaled_idx[bad_mask] = limit

    # Get the counts and reshape to multi-dim
    counts = np.bincount(scaled_idx.ravel(),minlength=limit+1)[:-1]
    counts.shape = data.shape[:-1] + (n_bins,)
    return counts,bins

def make_T2E_coeff(ret,outer=True):
    '''
    Written by Jordan O'Neal
    Creates coefficients needed for time-of-flight to kinetic energy conversion of electron spectrometer
    '''
    
    #t0 is prompt time (ionization time) determined from scattered light signal on MCPs
    #t0=1.9084660693575154e-01 #from lW8819
    t0_inner = 6.0155 #from lY7220, for inner anode
    t0_outer = 0.248 #from LY7220
    
    #this is from SIMION simulations
    coeff_mat = np.array([[-0.04214, -0.081895, 0.0020104, -2.0423e-05, 1.0489e-07, -2.8749e-10, 4.0119e-13, -2.2397e-16],
                      [11.707, -0.022102, 0.0001067, -2.4797e-07,1.9946e-10,0,0,0],
                      [-0.014009, -0.00020018, 1.8492e-05, -1.3306e-07,3.3689e-10,-2.8231e-13,0,0],
                      [0.00048195, 2.0962e-05, -7.3097e-07,4.9333e-09,-1.2199e-11,1.0086e-14,0,0],
                      [-5.4731e-06, -2.5076e-07, 7.5196e-09,-4.9729e-11,1.2178e-13,-1.0006e-16,0,0]])
    
    if outer:
        A_coeff = [t0_outer]
    else:
        A_coeff = [t0_inner]
    
    for i in range(coeff_mat.shape[0]):
        A_coeff.append(np.polyval(coeff_mat[i,::-1], ret))
    return A_coeff

def logic_win(X,win):
    '''
    From Greg McCracken
    '''
    I = (X >= win[0]) & (X < win[1])
    return I

def make_bins(xlim,dx):
    '''
    From Greg McCracken
    '''
    Edges_x = np.arange(xlim[0],xlim[1]+dx,dx)
    Bins_x = Edges_x[:-1]+dx/2
    return Edges_x,Bins_x

def analyze_fzp(fzp_b,dcom=400,bgs=False):
    '''
    Adapted from Greg McCracken
    Needed for center-of-mass determination of FZP spectrometer signal
    '''
    #dcom is full width of window around which to calculate COM
    
    #number of bins
    db = 1024//fzp_b.shape[1]
    
    #rebin
    pe,pb = make_bins((-0.5,1023.5),db)
    Nb = len(pb)
    N = fzp_b.shape[0]
        
    #subtract baseline
    if bgs:
        dp_bg = (400/db) #shift of window from max index
        bg_win = np.round(100/db) #half width of window
        mod = np.round(1024/db)
        for ii in range(N):
            imax = np.argmax(fzp_b[ii])
            cenm = int((imax - dp_bg)%mod)
            lower = int((cenm-bg_win)%mod)
            upper = int((cenm+bg_win)%mod)
            if upper < lower:
                base = np.concatenate([fzp_b[ii,lower:],fzp_b[ii,:upper]])
            else:
                base = fzp_b[ii,lower:upper]
            fzp_b[ii] -= base.mean()
    
    #find com
    
    dr = np.round(dcom/2)
    fzp_sum = np.zeros(N)
    fzp_com = np.zeros(N)
    for ii in range(N):
        imax = np.argmax(fzp_b[ii])
        cen = pb[imax]
        com_r = (cen-dr,cen+dr) 
        Icom = logic_win(pb,com_r)
        fzp_sum[ii] = fzp_b[ii].sum()
        fzp_com[ii] = (fzp_b[ii][Icom]*pb[Icom]).sum()/fzp_b[ii][Icom].sum()
        
    return fzp_sum, fzp_com

def dly_convert(runs,import_table=False,**kwargs):
    
    '''
    Creates new array of delays based on delay values from simulation matched to delay values form logbook 
    '''
    
    #check if needs to create delay table
    if import_table:
    
        delays_f_rf, runs_input_rf, delays_f_rt, runs_input_rt, run_dict = get_XAS_runs_LY72_corrected(max_delay_diff=1e-3, compute_delay_ver='v2')
        dlys = np.concatenate((delays_f_rt, delays_f_rf))
        runs_input = runs_input_rt + runs_input_rf
        
        dly_table = {}
        
        #checks if delay is already in table and only needs run added or table needs a new entry
        for dly,dly_runs in zip(dlys,runs_input):
    
            if np.isin(dly,list(dly_table.keys())):
                dly_table[dly] = np.concatenate([dly_table[dly],dly_runs])

            else:
                dly_table[dly] = dly_runs
            
    else:
        try:
            dly_table = kwargs['dly_table']
        except KeyError:
            print('Delay table not provided')
            return
    
    converted_dlys = np.zeros(len(runs))
    
    #create converted delays
    for i,run in enumerate(runs):
    
        for dly,dly_runs in zip(dly_table.keys(),dly_table.values()):

            if np.isin(run,dly_runs).any():
                
                converted_dlys[i] = dly
                
            if np.isin(run,np.array([109, 110, 111])).any():   #109-111 runs not loaded into get_XAS_runs_LY72.py. Delay calibrated based on runs 131-133
                converted_dlys[i] = 8.98
                
    converted_dlys[converted_dlys == 0] = 0
            
    return converted_dlys    

def laplacian1D_S(N):
    '''
    From spooktroscopy package
    Calculate Laplacian matrix of size N
    '''
    Lmat = sps.eye(N)*(-2)
    if N > 1:
        b = np.ones(N-1)
        Lmat += sps.diags(b, offsets=1) + sps.diags(b, offsets=-1)
    return Lmat

def L_curve_search(AtA,Atb,btb,x0 = -10, x3 = 8, epsilon = 1e-3,method='golden',smooths=np.logspace(-3,6,100)):
    
    '''
    Optimization of hyperparameters for Tikhonov regularization
    Uses either the algorithm from DOI 10.1088/2633-1357/abad0d or point of maximum curvature
    Algorithm "golden" is suggested
    '''
    
    if method == 'golden':
        
        '''
        This algorithm uses four initial points
        -two outer points from user input, two inner poitns from golden ratio
        It uses points 1-3 and points 2-4 to calculate curvatures
        Then it narrows search range based on curvatures and golden ratio until threshold is reached
        '''
        
        #golden ratio
        phi = (1+np.sqrt(5))/2
        
        #four hyperparameters selected from golden ratio based on initial input ranges (x0,x3)
        l = np.zeros(4)

        l[0] = 10**x0
        l[3] = 10**x3
        l[1] = 10**((x3+phi*x0) / (1+phi))
        l[2] = 10**(x0 + (x3-np.log10(l[1])))

        P = np.zeros((4,2))
        
        #Initialize algorithm to solve Ax=b
        spokane = spook.SpookPosL1(Atb, AtA, "contracted",lsparse=0,lsmooth=(10**x0,10**x0))
        
        #Solve for hyperparemter x0
        spokane.solve(0,(10**x0,10**x0))
        X = spokane.getXopt().flatten()                                
        
        #Calculate L2 residual for solution
        L2 = spokane.residueL2(btb)**2
        
        #Calculate smoothness of solution based on laplacian
        smoothness = np.square(laplacian1D_S(len(X)) @ X)[1:-1].sum()
        
        #l_r, u_x, u_r, l_x are the maxima and minima of search range for smoothness and L2
        l_r = np.min(np.log10(L2))
        u_x = np.max(np.log10(smoothness))
        
        spokane.solve(0,(10**x3,10**x3))
        X = spokane.getXopt().flatten()                                
        # res = A_split @ X - b_split
        # L2 = np.linalg.norm(res,2)**2
        L2 = spokane.residueL2(btb)**2
        smoothness = np.square(laplacian1D_S(len(X)) @ X)[1:-1].sum()
        
        u_r = np.max(np.log10(L2))
        l_x = np.min(np.log10(smoothness))
        
        #use extrema of search range to rescale results to (-10,10) on both axes to avoid machine precision errors
        def L2_new(r):

            return (20/(u_r-l_r))*(np.log10(r) - (10*l_r + 10 *u_r)/(20))
        
        def sm_new(x):
            
            return (20/(u_x-l_x))*(np.log10(x) - (10*l_x + 10 *u_x)/(20))
    
        #calculate L2 residual and smooth for four initial points
        for i in range(4):

            spokane.solve(0,(l[i],l[i]))
            X = spokane.getXopt().flatten()                                
            # res = A_split @ X - b_split
            # L2 = np.linalg.norm(res,2)**2
            L2 = spokane.residueL2(btb)**2
            
            smoothness = np.square(laplacian1D_S(len(X)) @ X)[1:-1].sum()

            P[i,0] = L2_new(L2)
            P[i,1] = sm_new(smoothness)

        iterations = 0

        save = np.empty((0,7))

        #restrict search zone according to golden ratio and curvatures until error threshold is reached
        while (l[3]-l[0])/l[3] > epsilon:
            
            #Calculate curvatures
            C1 = menger(P[:3])
            C2 = menger(P[1:])
            
            #save iteraction values for debugging
            if iterations == 0:
                save = np.array([[l[0],l[1],l[2],l[3],C1,C2,iterations]])
            else:
                step = np.array([[l[0],l[1],l[2],l[3],C1,C2,iterations]])
                save = np.append(save,step,axis=0)

    #         print(iterations)
    #         print(l)
    #         print(C1,C2)
    #         print(P)
    #         print()
            
            #first shift right side in to get away from negative curvature point
            #point we want is the positive curvature "elbow"
            while (C2 < 0):

                l[3] = l[2]
                P[3] = P[2]
                l[2] = l[1]
                P[2] = P[1]

                l[1] = 10**((np.log10(l[3])+phi*np.log10(l[0])) / (1+phi))

                spokane.solve(0,(l[1],l[1]))
                X = spokane.getXopt().flatten()                                
                # res = A_split @ X - b_split
                # L2 = np.linalg.norm(res,2)**2
                L2 = spokane.residueL2(btb)**2
                smoothness = np.square(laplacian1D_S(len(X)) @ X)[1:-1].sum()

                P[1,0] = L2_new(L2)
                P[1,1] = sm_new(smoothness)

                C2 = menger(P[1:])
            
            #iterate left and right side towards each other until search range is small enough
            if C1 > C2:

                lMC = l[1].copy()

                l[3] = l[2]
                P[3] = P[2]
                l[2] = l[1]
                P[2] = P[1]

                l[1] = 10**((np.log10(l[3])+phi*np.log10(l[0])) / (1+phi))

                spokane.solve(0,(l[1],l[1]))
                X = spokane.getXopt().flatten()                                
                # res = A_split @ X - b_split
                # L2 = np.linalg.norm(res,2)**2
                L2 = spokane.residueL2(btb)**2
                smoothness = np.square(laplacian1D_S(len(X)) @ X)[1:-1].sum()

                P[1,0] = L2_new(L2)
                P[1,1] = sm_new(smoothness)

            else:
                lMC = l[2].copy()
                l[0] = l[1]
                P[0] = P[1]
                l[1] = l[2]
                P[1] = P[2]

                l[2] = 10**(np.log10(l[0]) + (np.log10(l[3]) - np.log10(l[1])))

                spokane.solve(0,(l[2],l[2]))
                X = spokane.getXopt().flatten()                                
                # res = A_split @ X - b_split
                # L2 = np.linalg.norm(res,2)**2
                L2 = spokane.residueL2(btb)**2
                smoothness = np.square(laplacian1D_S(len(X)) @ X)[1:-1].sum()

                P[2,0] = L2_new(L2)
                P[2,1] = sm_new(smoothness)

            iterations += 1

        return lMC, iterations, save, L2_new, sm_new
    
    elif method == 'curvature':
        
        #this method calculates curvatures for a number of hyperparameters and selects the one that is greatest

        spokane = spook.SpookPosL1(b, A, "raw",lsparse=0,lsmooth=(smooths[0],smooths[0]))
        
        L2_fit = np.zeros(len(smooths))
        smoothness_fit = np.zeros(len(smooths))
        
        for jdx,j in enumerate(smooths):
            spokane.solve(0,(j,j))
            X = spokane.getXopt().flatten()                                
            # res = A_split @ X - b_split
            # L2 = np.linalg.norm(res,2)**2
            spokane.residueL2(b.T @ b)**2
            smoothness = np.square(laplacian1D_S(len(X)) @ X)[1:-1].sum()
            
            L2_fit[jdx] = L2_new(L2)
            smoothness_fit[jdx] = sm_new(smoothness)
        
        cs_res = splrep(smooths,L2_fit)
        cs_lap = splrep(smooths,smoothness_fit)
        
        curvature = np.multiply(splev(smooths,cs_res,1),splev(smooths,cs_lap,2)) - np.multiply(splev(smooths,cs_lap,1),splev(smooths,cs_res,2))
        curvature /= np.power(np.square(splev(smooths,cs_res,1))+np.square(splev(smooths,cs_lap,1)),1.5)
        
        lMC = smooths[np.argmax(curvature)]
        
        return lMC
    
    # elif method == 'pruning':

def menger(P):
    
    '''
    Calculate Menger curvature of three points
    '''

    C = 2*(P[0,0]*P[1,1] + P[1,0]*P[2,1] +  P[2,0]*P[0,1] - P[0,0]*P[2,1] - P[1,0]*P[0,1] - P[2,0]*P[1,1])

    P01 = np.square(P[1,0]-P[0,0]) + np.square(P[1,1]-P[0,1])
    P12 = np.square(P[2,0]-P[1,0]) + np.square(P[2,1]-P[1,1])
    P21 = np.square(P[0,0]-P[2,0]) + np.square(P[0,1]-P[2,1])

    C /= np.sqrt(P01*P12*P21)

    return C      

def get_XAS_runs_LY72_corrected(min_num_runs=2, max_delay_diff=0.1, compute_delay_ver='v1'):
    
    '''
    Written by Erik Isele
    Converts logbook delays to simulated delays using simulations
    Chicane delay is correct in logbook
    Undulator delay needs to be calculated
    '''
    
    runs_tot = {}
    
    #dictionary with logbook delay, number of undulators, drift delay, reamplified or not, chicane delay as keys
    # values are respective runs
    
    # nominal delays
    runs_tot['1.25,  5U, 0d, RF, C0'] = np.array([86])
    # runs_tot['0.0 , prOnly'] = np.arange(231, 251)
    runs_tot['3.75,  5U, 0d, RF, C2.5'] = np.array([87, 88, 89, 90])
    runs_tot['2.5 ,  6U, 0d, RF, C1'] = np.array([91, 92, 93, 116, 117, 118]) 
    runs_tot['3.25,  6U, 0d, RF, C1.75'] = np.array([94, 95, 96, 119, 120, 121])
    runs_tot['4.0 ,  6U, 0d, RF, C2.5'] = np.array([97, 98, 99, 122, 123, 124]) 
    runs_tot['4.75,  6U, 0d, RF, C3.25'] = np.array([100, 101, 102, 125, 126, 127]) 
    runs_tot['5.5 ,  6U, 0d, RF, C4'] = np.array([106, 107, 108, 128, 129, 130]) 
    runs_tot['1.5 ,  6U, 0d, RF, C0'] = np.array([113, 114, 115, 134, 135, 136])
    runs_tot['6.25,  6U, 0d, RF, C4.75'] = np.array([131, 132, 133])
    runs_tot['7.0 , 10U, 0d, RF, C4.5'] = np.array([137, 138, 139, 146, 147, 148])
    runs_tot['9.0 , 10U, 0d, RF, C6.5'] = np.array([140, 141, 142, 143, 144, 149, 150, 151])
    runs_tot['0.5 ,  2U, 0d, RT, C0'] = np.array([161, 162, 163, 164, 165, 166, 167])
    runs_tot['0.75,  3U, 0d, RT, C0'] = np.array([172, 173, 174, 178, 179, 180])
    runs_tot['1.25,  5U, 0d, RT, C0'] = np.array([168, 169, 170, 171, 175, 176, 177]) 
    runs_tot['1.25,  5U, 0d, RF, C0'] = ([181, 182, 183, 184]) 
    
    # compute new delays from runtable values
    new_delays_rf = []
    new_delays_rt = []
    run_dict = {}
    for k in runs_tot.keys():
        #pull number of undulators from dictionary key
        numU = int(k[6:8])
        #pull drift delay from dictionary key
        numD = int(k[11:12])
        #reamplified or not
        reamplification = (str(k[15:17])=='RT')
        #chicane delay
        delC = float(k[20:])
        if numD > 0: # Outdated values from lw88
            driftK = 3.276
        else: driftK = None
        
        for run in runs_tot[k]:
            run_dict[run] = {}
            run_dict[run]['numU'] = numU
            run_dict[run]['delC'] = delC
        
        
        if not reamplification:
            if compute_delay_ver == 'v1':
                new_delays_rf.append([compute_delay(numU, numD, delC, driftK), runs_tot[k]])
            elif compute_delay_ver == 'v2':
                new_delays_rf.append([compute_delay_v2(numU, numD, delC, driftK), runs_tot[k]])
            else:
                print('select v1 or v2')
                
        if reamplification:
            if compute_delay_ver == 'v1':
                new_delays_rt.append([compute_delay(numU, numD, delC, driftK), runs_tot[k]])
            elif compute_delay_ver == 'v2':
                new_delays_rt.append([compute_delay_v2(numU, numD, delC, driftK), runs_tot[k]])
            else:
                print('select v1 or v2')

    # group similar delays together
    new_delays_grouped_rf = []
    for i in range(len(new_delays_rf)):

        if i == 0:
            new_delays_grouped_rf.append(new_delays_rf[i])
        else:
            # decide whether to add new delay or only add runs
            new = True
            for k in range(i):
                if np.abs(new_delays_rf[i][0] - new_delays_rf[k][0]) / new_delays_rf[i][0] < max_delay_diff: # set to 0.01 to separate all different delays
                    new = False

            if new:
                new_delays_grouped_rf.append(new_delays_rf[i])
            else: # concatenate runs into grouped delay
                idx = np.argmin(np.abs(new_delays_rf[i][0] - np.array(new_delays_grouped_rf, dtype=object)[:,0]))
                new_delays_grouped_rf[idx][1] = list(new_delays_grouped_rf[idx][1]) + list(new_delays_rf[i][1])

    # sort delays
    idx = np.argsort(np.array(new_delays_grouped_rf, dtype=object)[:,0])
    new_delays_sorted_rf = np.array(new_delays_grouped_rf, dtype=object)[idx]
    
    # strip out delays with too few runs
    delays_f_rf, runs_input_rf = [], []
    for i in range(new_delays_sorted_rf.shape[0]):
        if len(new_delays_sorted_rf[i][1]) >= min_num_runs:
            delays_f_rf.append(np.round(new_delays_sorted_rf[i][0], 2))
            runs_input_rf.append(new_delays_sorted_rf[i][1])
            
    delays_f_rf = np.array(delays_f_rf)
    
    # group similar delays together
    new_delays_grouped_rt = []
    for i in range(len(new_delays_rt)):

        if i == 0:
            new_delays_grouped_rt.append(new_delays_rt[i])
        else:
            # decide whether to add new delay or only add runs
            new = True
            for k in range(i):
                if np.abs(new_delays_rt[i][0] - new_delays_rt[k][0]) / new_delays_rt[i][0] < max_delay_diff: # set to 0.01 to separate all different delays
                    new = False

            if new:
                new_delays_grouped_rt.append(new_delays_rt[i])
            else: # concatenate runs into grouped delay
                idx = np.argmin(np.abs(new_delays_rt[i][0] - np.array(new_delays_grouped_rt, dtype=object)[:,0]))
                new_delays_grouped_rt[idx][1] = list(new_delays_grouped_rt[idx][1]) + list(new_delays_rt[i][1])

    # sort delays
    idx = np.argsort(np.array(new_delays_grouped_rt, dtype=object)[:,0])
    new_delays_sorted_rt = np.array(new_delays_grouped_rt, dtype=object)[idx]
    
    # strip out delays with too few runs
    delays_f_rt, runs_input_rt = [], []
    for i in range(new_delays_sorted_rt.shape[0]):
        if len(new_delays_sorted_rt[i][1]) >= min_num_runs:
            delays_f_rt.append(np.round(new_delays_sorted_rt[i][0], 2))
            runs_input_rt.append(new_delays_sorted_rt[i][1])
            
    delays_f_rt = np.array(delays_f_rt)
    
    run_ret, run_pitch, run_order = 400, 8.45, 1
    
    return delays_f_rf, runs_input_rf, delays_f_rt, runs_input_rt, run_dict

def compute_delay_v2(numU, numD=0, delC=0., driftK=None): 
    
    '''
    From Erik Isele
    Calculate actual delay based on number of undulators, chicane delay, and if reamplified
    '''
    
    # need driftK only if there are drift undulators
    t0_chicane = 0.08 # fs
    T_520eV = 0.00795 # fs
    L3_energy = 4075. # MeV, only for drift undulators
    c = 3e8
    
    if numD == 0:
        if delC<1e-3:
            return t0_chicane + numU*T_520eV*87*(1/3)
        else:
            return t0_chicane + numU*T_520eV*87*(3/3) + delC
    else:
        lamU = 0.039 # m
        gamma = L3_energy / 0.511 # MeV to gamma
        lam = lamU / (2*gamma**2) * (1 + driftK**2/2)
        T_drift = lam / c * 1e15
        if delC < 1e-3:
            return t0_chicane + numU*T_520eV*87*(1/3) + numD*T_520eV*87
        else:
            return t0_chicane + numU*T_520eV*87*(3/3) + numD*T_520eV*87 + delC
        