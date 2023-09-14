This is analysis code for attosecond pump, attoscond pulse x-ray absorption experiments at the LCLS. It uses the covariance based regression method [Spectral Domain Ghost Imaging](https://github.com/congzlwag/spook) (spooktroscopy) to solve $Ax=b$ under the Tikhonov regularization $\lambda|Lx|_2^2$ where $A$ is a pixelated spectrometer reading for each XFEL shot, $b$ is a bucketed ionized electron counts for each shot, and $\lambda$ is the hyperparameter for regularization. The code is broken into two segments:
1. [dataset_creation_v2.py](/dataset_creation_v2.py) which takes preprocessed electron detector and spectrometer data for various pump,probe delays and concatenates further preprocesses it to create A and b matrices. This script is meant to run in parallel on a computing cluster (see [dataset_creation.sh](/dataset_creation.sh)
2. [xas_parent.py](/xas_parent.py) is a class to perform the analysis. The class consists of the following main and optional functions which should initially be run in order:
    1. init: loads in normalization arrays, initial preset analysis parameters, and the different delays and run #s for each delay
    2. load_data: Loads in A and b matrices, pump and probe energies from input path (previously saved with [dataset_creation_v2.py](/dataset_creation_v2.py). This also loads in bpmx values but must concatenate for all runs in a delay and hence is timely. (This concatenation should be added to dataset_creation in the future)
    3. create_pump_bins: Bins A and b matrices according to shot pump energies
    4. create_bpmx_bins: Bins A and b matrices according to beam position monitor along x-axis
    5. create_bpmx_pump_means: Calculates the mean beam position and pump energy for each bin
    6. calc_sigma: Calculates # of shots in each bin to be used for weighting in averaging (In the future steps iii-vi should be combined to one function)
    7. L_curve_select: Automatically selects the smoothness hyperparemter for regularization.
    8. alpha_scan: Automatically calculates a scaling parameter alpha such that (alpha*pump_energy = 3w valence electrons) to remove contaminate signal from third harmonic of the pump pulse.
    9. calculate_X: Performs spooktroscopy on the A and corrected b matrices for each delay using the hyperparameter from vii and calculates the average absorption in the three regions of interest (on resonance, off resonance, near resonance)
    10. fit_X: Uses linear regression to calculate cresonant absorption vs pump energy for each delay and plots it

The analysis code also contains additional optional functions:
1. masks.create_edge_mask: Creates masks for each A and b matrix removing shots that are cut by the edges of the spectrometer
2. mask_data: Applies the edge mask
3. rebin_data: Rebins matrices to smaller bins. Bins do __not__ need to be an integer multiple of the current bin size (note previous data is not saved so you cannot go to larger bins)
4. rebin_yag: Rebins the spectrometer Ce:YAG scintillator crystal transmission function according to transfer function from 3
5. set_params: Change the analysis parameters (nbins, pump_bins, pump_max, bpmx_bins, bpmx_min, bpmx_max) via keyword inputs
6. plot_A: Plots the average A matrix for each delay

Note that some functions need the outputs of previous functions (e.g. calculate_X needs alpha and lambda from steps vii and viii), and hence when changing parameters or rebinning, these inputs should be recalculated. However, the entire analysis does not need to be rerun for each change.

See the code comments for instructions on each function input and structure. Below are select plots to demonstrate some functions. _Note due to the unpublished nature of this research project, not all step outputs can be plotted_. 
  
