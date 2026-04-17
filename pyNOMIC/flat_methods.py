#----------------------------------------
# IMPORTS
#----------------------------------------

import numpy as np
import os, psutil
from astropy.io import fits
from tqdm.auto import tqdm
from scipy.optimize import curve_fit
from scipy.interpolate import NearestNDInterpolator
from astropy.convolution import convolve_fft,  Ring2DKernel, Gaussian2DKernel
from multiprocessing.pool import ThreadPool as Pool
import pyNOMIC.helper_functions as hf

#----------------------------------------
# CLASSES
#----------------------------------------

class PSFSubtraction(object):

    """
    Tweaked image integration to allow for masking of the target star for constructing a flat.
    """

    def __init__(self, params):
        
        """
        Parameters (contained inside a tuple):
        ----------------------
        psf_subtracted_dir: string/Path object
            Directory where images will be saved
        files: list or array 
            List of raw file paths, sorted 
        chops: string array
            List of chop states corresponding to the file list, entries are either "CHOP_A" or "CHOP_B"
        maxima: float tuple array
            Tuples encoding location of the PSF in the images
        tempflat: 2D numpy array
            Temporary flat applied to locate the star
        badmap: 2D image array
            Bad pixel map, where bad pixels are set to 0 and all other pixels are set to 1.
        nbg: integer
            Number of frames to use in rolling background subtraction
        edge_cut: integer
            Number of pixels to remove at the edges of images
        smooth: integer
            Radius of smoothing kernel, divided by 5
        windowsize: integer
            Half width/height of the cutout image (which is 1:1 aspect ratio)
        remove_residual: boolean
            Enables removal of psf residual through highpass filtering. Disabled by default.
        """
        
        self.params = params
    
    def __call__(self, i):

        """
        Parameters:
        ----------------------
        indices: integer list or array
            Group of file indices to process from 'files'

        Returns: 
        ---------------------- 
        maximum[0]: float
            X-axis location of the maximum
        maximum[1]: integer
            Y-axis location of the maximum
        """
        
        psf_subtracted_dir, files, chops, maxima, flat, badmap, nbg, edge_cut, smooth, windowsize, remove_residual = self.params

        # Locate adjacent images of the same chop state
        bg_indices = np.arange(i - 2*nbg + 1, i + 2*nbg, 2)

        # Exclude nonexistent indices and the current index
        bg_indices = bg_indices[(bg_indices >= 0) & (bg_indices < len(files)) & (bg_indices != i)]

        # Open image and get array shape
        unsubtracted = fits.open(files[i])
        img = unsubtracted[0].data[0]
        array_shape = np.shape(unsubtracted[0].data[0])
        
        # Create background image from adjacent images (different chop state)
        bg = np.zeros(array_shape)
        for j in bg_indices:
            hdul = fits.open(files[j])
            bg += hdul[0].data[0]
            hdul.close()
            
        # Averaged background
        bg = bg / len(bg_indices)

        # Subtract background
        subtracted_frame = (img - bg)

        if flat is not None:
            # Apply flat correction
            subtracted_frame = np.nanmedian(flat)*subtracted_frame/flat

        if badmap is not None:
            # Create boolean map from badmap to remove bad pixels
            subtracted_frame[badmap < 1] = np.nan

        # Cut out edges, which are often dead columns/rows
        subtracted_frame = subtracted_frame[edge_cut:-1*edge_cut,edge_cut:-1*edge_cut]

        subtracted_frame = np.pad(subtracted_frame, edge_cut, mode='edge')
        
        # Subtract convolved background
        bg_model = convolve_fft(np.pad(subtracted_frame, 10*smooth, mode='edge'), Ring2DKernel(5*smooth, 4*smooth))[10*smooth:-10*smooth, 10*smooth:-10*smooth]

        nx = np.linspace(0, array_shape[1]-1, array_shape[1])
        ny = np.linspace(0, array_shape[0]-1, array_shape[0])
        nx, ny = np.meshgrid(nx, ny)
        
        wx = np.linspace(0, 2*windowsize-1, 2*windowsize)
        wy = np.linspace(0, 2*windowsize-1, 2*windowsize)
        wx, wy = np.meshgrid(wx, wy)

        for j in range(2):
        
            new_frame = subtracted_frame - bg_model
        
            cutout = new_frame[int(maxima[i][0]-windowsize):int(maxima[i][0]+windowsize),
                                int(maxima[i][1]-windowsize):int(maxima[i][1]+windowsize)]
            
            mask = np.where(~np.isnan(cutout))
            interp = NearestNDInterpolator(np.transpose(mask), cutout[mask])
            cutout = interp(*np.indices(cutout.shape))
            
            # Run curve_fit to get airy best fit parameters
            reffit, _ = curve_fit(hf.airydisk_ravel, (wx, wy), cutout.ravel(), p0=[np.max(cutout),\
                        30,30, -1, 0, windowsize, windowsize],\
                       bounds=([0, 1, 1, 1*-np.inf, 0, 1, 1], [10*np.max(cutout), 200, 200, np.inf, 2*np.pi, 2*windowsize, 2*windowsize]))
            
            psf_model = hf.airydisk((nx, ny), reffit[0], reffit[1], reffit[2], reffit[3], reffit[4],\
                                            int(maxima[i][1])+reffit[5]-windowsize, int(maxima[i][0])+reffit[6]-windowsize) 
            
            bg_model = convolve_fft(subtracted_frame - psf_model, Ring2DKernel(3*smooth, 2*smooth))

        if remove_residual == True:
            
            residual = (subtracted_frame - psf_model)
            residual_bg = convolve_fft(residual, Gaussian2DKernel(2))
            
            radius = 1.2*int(np.sqrt(reffit[1]**2 + reffit[2]**2))
            psfrem = hf.psf_removal_mask(( int(maxima[i][1])+reffit[5]-windowsize, int(maxima[i][0])+reffit[6]-windowsize), radius, 1.2*radius, array_shape[1], array_shape[0])
    
            cirmask = hf.circular_mask(( int(maxima[i][1])+reffit[5]-windowsize, int(maxima[i][0])+reffit[6]-windowsize), radius, array_shape[1], array_shape[0]) ^\
                        hf.circular_mask(( int(maxima[i][1])+reffit[5]-windowsize, int(maxima[i][0])+reffit[6]-windowsize), 1.2*radius, array_shape[1], array_shape[0])

            if flat is not None:
                psf_model = psf_model*flat/np.nanmedian(flat)
    
            final = (img - psf_model)*(1-psfrem)+(img - psf_model - residual_bg + np.median(residual_bg[cirmask]) )*psfrem
        else:
            if flat is not None:
                psf_model = psf_model*flat/np.nanmedian(flat)
            final = img - psf_model

        unsubtracted.close()
        
        # Write image to file
        newhdul = fits.HDUList([fits.PrimaryHDU(data=(final))])
        newhdul.writeto(os.path.join(psf_subtracted_dir, "psfsubtracted_"+files[i].name), overwrite=True)
        newhdul.close()
    
        return int(maxima[i][0])+reffit[6]-windowsize, int(maxima[i][1])+reffit[5]-windowsize

#----------------------------------------
# FUNCTIONS
#----------------------------------------

def create_badmap(files, tolerance=0.9, sigma=.4, edge_cut=3, smooth=30,threadcount=50):

    """
    Create a badmap by constructing a crude flat, running it through a high pass filter, and masking out large deviations.
    
    Parameters:
    ----------------------
    files: list or array 
        List of raw file paths to stack.
    tolerance (optional): float
        Fraction of available memory to be used for integration.
    sigma (optional): float
        Number of standard deviations to include in the badmap.
    edge_cut (optional): integer
        Number of pixels to remove from the edges of the image before high pass filtering. Default value is 3 pixels.
    smooth (optional): integer
        Outer radius of the smoothing kernel. Default value is 30 pixels.
    threadcount (optional): integer
        Number of threads to employ in multithreading. Default value is 50 threads.
    
    Returns: 
    ---------------------- 
    flat: 2D image array
        Stacked flat frame.
        
    badmap: 2D image array
        Bad pixel map, where bad pixels are set to 0 and all other pixels are set to 1.

    filtered_frame: 2D image array
        High pass filtered flat from which the badmap was created.
    """
    
    #v Check available memory
    stats = psutil.virtual_memory()  # returns a named tuple
    available = float(getattr(stats, 'available'))

    # Open a file and check file_size and image shape
    hdul = fits.open(files[0])
    file_size = float(hdul[0].data[0].nbytes)
    array_shape = np.shape(hdul[0].data[0])
    hdul.close()

    # Calculate memory buffer based on available memory, size of files, and tolerance
    buffer = int(np.ceil((file_size*threadcount*len(files))/(tolerance*available)))

    print("Creating integrated files for correlation...")
    print("Using a buffer of ", int(len(files)/buffer), " frames...")

    # Split files into buffers
    splitlist = np.linspace(0, len(files), 1+buffer)[1:-1].round().astype(int)
    filebufs = np.split(files, splitlist)

    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        bigarr, filecounts = zip(*tqdm(pool.imap(hf.IntegrateFrames((array_shape)), filebufs),\
                                           desc="integrating files", total=len(filebufs)))

    # Create flat by mean of the images
    flat = np.sum(bigarr, axis=0)/np.sum(filecounts)

    # Remove zero values from the flat, replace with the minimum value
    flat[flat == 0] = np.min(flat[flat != 0]) 
    smooth_buf = int(1.25*smooth)

    # Create copy image and pad it for convolution
    new_image = np.copy(flat)
    new_image = np.pad(new_image[edge_cut:-1*edge_cut, edge_cut:-1*edge_cut], smooth_buf+edge_cut, mode='edge')

    # High pass filter
    filtered_frame = flat - convolve_fft(new_image, Ring2DKernel(smooth, int(0.8*smooth)))[smooth_buf:-1*smooth_buf, smooth_buf:-1*smooth_buf]
    #filtered_frame[~bools] = np.nan

    # Create badmap by setting pixels above threshold to 0
    badmap = np.ones(np.shape(filtered_frame))
    badmap[(filtered_frame > (sigma*np.std(filtered_frame)+np.median(filtered_frame))) |\
            (filtered_frame < (-1*sigma*np.std(filtered_frame)+np.median(filtered_frame)))] = 0

    return flat, badmap, filtered_frame

def create_stacked_flat(files, chops, chop_direction="UP-DOWN", tolerance=0.9, threadcount=50):
    
    """
    Creates a stacked flat frame for each chop state. They are either combined into a single flat or returned separately.
    
    Parameters:
    ----------------------
    files: list or array 
        List of raw file paths to stack.
    chops: list or array
        List of chop states corresponding to the file list, entries are either "CHOP_A" or "CHOP_B".
    chop_direction (optional):
        The chopping direction employed, either "UP-DOWN", "LEFT-RIGHT", or "SEPARATE" if separate flats for each chop are desired.
        Default value is "UP-DOWN", used for single-sided imaging.
    tolerance (optional): float
        Fraction of available memory to be used for integration. Default value is 0.9.
    threadcount (optional): integer
        Number of threads to employ in multithreading. Default value is 50 threads.
    
    Returns: 
    ---------------------- 
    flat: 2D image array
        Stacked flat frame.

    OR

    chopa_mean_frame: 2D image array
        Stacked flat frame for CHOP_A
    
    chopb_mean_frame:   
        Stacked flat frame for CHOP_B
    """

    # Check available memory
    stats = psutil.virtual_memory()  # returns a named tuple
    available = float(getattr(stats, 'available'))

    # Open a file and check file_size and image shape
    hdul = fits.open(files[0])
    file_size = float(hdul[0].data.nbytes)
    array_shape = np.shape(hdul[0].data)
    hdul.close()

    # Force array shape to have correct dimensions
    if len(array_shape) == 3:
        array_shape = (array_shape[1], array_shape[2])

    # Separate files into chops
    chopa_files = files[chops == "CHOP_A"]
    chopb_files = files[chops == "CHOP_B"]

    # Calculate memory buffers based on available memory, size of files, and tolerance
    a_buffer = int(np.ceil((file_size*threadcount*len(chopa_files))/(tolerance*available)))
    b_buffer = int(np.ceil((file_size*threadcount*len(chopb_files))/(tolerance*available)))

    print("Creating integrated files for correlation...")
    print("Using a buffer of ", int(len(chopa_files)/a_buffer), " frames...")
    
    # Split files into buffers
    a_splitlist = np.linspace(0, len(chopa_files), 1+a_buffer)[1:-1].round().astype(int)
    a_filebufs = np.split(chopa_files, a_splitlist)
    
    b_splitlist = np.linspace(0, len(chopb_files), 1+b_buffer)[1:-1].round().astype(int)
    b_filebufs = np.split(chopb_files, b_splitlist)

    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        a_bigarr, a_filecounts = zip(*tqdm(pool.imap(hf.IntegrateFrames((array_shape)), a_filebufs),\
                                           desc="integrating files", total=len(a_filebufs)))

    # Create chopa flat by mean of the images
    chopa_mean_frame = np.sum(a_bigarr, axis=0)/np.sum(a_filecounts)
    
    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        b_bigarr, b_filecounts = zip(*tqdm(pool.imap(hf.IntegrateFrames((array_shape)), b_filebufs),\
                                           desc="integrating files", total=len(b_filebufs)))

    # Create chopb flat by mean of the images
    chopb_mean_frame = np.sum(b_bigarr, axis=0)/np.sum(b_filecounts)

    # Stitch chops together such that the halves of the image without the target are combined together
    if chop_direction == "UP-DOWN":

        flat = np.concatenate((chopb_mean_frame[:int(array_shape[0]/2), :], chopa_mean_frame[int(array_shape[0]/2):, :]))

    elif chop_direction == "LEFT-RIGHT":

        flat = np.concatenate((chopb_mean_frame[:, :int(array_shape[1]/2)].T, chopa_mean_frame[:, int(array_shape[1]/2):].T)).T

    else:

        raise ValueError("Invalid chop direction")

    # Remove zero values from the flat, replace with the minimum value    
    flat[flat == 0] = np.min(flat[flat != 0]) 

    # Remove zero values from the flat, replace with the minimum value    
    chopa_mean_frame[chopa_mean_frame == 0] = np.min(chopa_mean_frame[chopa_mean_frame != 0]) 
    chopb_mean_frame[chopb_mean_frame == 0] = np.min(chopb_mean_frame[chopb_mean_frame != 0]) 

    return flat, chopa_mean_frame, chopb_mean_frame

def subtract_psfs(files, chops, maxima, flat=None, badmap=None, remove_residual=False, windowsize=30, smooth=5, edge_cut=2, nbg=1, threadcount=50, prefix=''):

    """

    Subtracts the stellar PSF from every image.
    
    Parameters (contained inside a tuple):
    ----------------------
    files: list or array 
        List of raw file paths, sorted 
    chops: string array
        List of chop states corresponding to the file list, entries are either "CHOP_A" or "CHOP_B"
    maxima: float tuple array
        Tuples encoding location of the PSF in the images
    flat(optional): 2D numpy array
        Temporary flat applied to locate the star
    badmap(optional): 2D image array
        Bad pixel map, where bad pixels are set to 0 and all other pixels are set to 1.
    nbg(optional): integer
        Number of frames to use in rolling background subtraction
    edge_cut (optional): integer
        Number of pixels to remove at the edges of images
    smooth (optional): integer
        Radius of smoothing kernel, divided by 5
    windowsize (optional): integer
        Half width/height of the cutout image (which is 1:1 aspect ratio)
    remove_residual (optional): boolean
        Enables removal of psf residual through highpass filtering. Disabled by default.
    threadcount (optional): integer
        Number of threads to employ in multithreading. Default value is 50 threads.
    prefix (optional): string
        Prefix to add to directory name when saving image.
    """
    
    print("Subtracting psfs....")

    # Create directory to save files
    root_dir = os.path.dirname(os.path.dirname(files[0]))
    psf_subtracted_dir = os.path.join(root_dir, prefix+'psfsubtracted')
    if not os.path.exists(psf_subtracted_dir):
        os.makedirs(psf_subtracted_dir)

    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        
        max_x, max_y = zip(*tqdm(pool.imap(PSFSubtraction((psf_subtracted_dir, files, chops, maxima, flat, badmap, nbg, edge_cut,
                                                             smooth, windowsize, remove_residual)), range(len(files))), total=len(files)))

    return psf_subtracted_dir 