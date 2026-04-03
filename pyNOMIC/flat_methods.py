#----------------------------------------
# IMPORTS
#----------------------------------------

import numpy as np
import os, psutil
from astropy.io import fits
from tqdm.auto import tqdm
from astropy.convolution import convolve_fft,  Ring2DKernel
from multiprocessing.pool import ThreadPool as Pool
from pyNOMIC.helper_functions import IntegrateFrames, psf_removal_mask

#----------------------------------------
# CLASSES
#----------------------------------------

class IntegrateFrames_Flat(object):

    """
    Tweaked image integration to allow for masking of the target star for constructing a flat.
    """

    def __init__(self, params):
        
        """
        Parameters (contained inside a tuple):
        ----------------------
        files: list or array 
            List of raw file paths, sorted 
        chops: string array
            List of chop states corresponding to the file list, entries are either "CHOP_A" or "CHOP_B"
        tempflat: 2D numpy array
            Temporary flat applied to locate the star
        nbg: integer
            Number of frames to use in rolling background subtraction
        array_shape: integer tuple
            Tuple containing image dimensions, from numpy.shape
        edge_cut: integer
            Number of pixels to remove at the edges of images
        inner_rad: integer
            Inner radius of the star mask, insde of which the star is fully replaced by the background
        outer_rad: integer
            Outer radius of the star mask
        """
        
        self.params = params
    
    def __call__(self, indices):

        """
        Parameters:
        ----------------------
        indices: integer list or array
            Group of file indices to process from 'files'

        Returns: 
        ---------------------- 
        stacked_img: 2D image array
            Integrated stack of frames (sum)
            
        count: integer
            Number of frames
        """
        
        files, chops, tempflat, nbg, array_shape, edge_cut, inner_rad, outer_rad = self.params
        
        # array for integrating files      
        buf_3D = np.zeros((len(indices), array_shape[0], array_shape[1]))
        
        count = 0
        
        for k in range(len(indices)):

            # Locate adjacent images of the same chop state
            bg_indices = np.arange(indices[k] - 2*nbg + 1, indices[k] + 2*nbg, 2)

            # Exclude nonexistent indices and the current index
            bg_indices = bg_indices[(bg_indices >= 0) & (bg_indices < len(files)) & (bg_indices != indices[k])]

            # Open image and get array shape
            unsubtracted = fits.open(files[indices[k]])
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

            # Subtract background and apply flat correction
            subtracted_frame = (unsubtracted[0].data[0] - bg)/tempflat

            # Cut out edges, which are often dead columns/rows
            subtracted_frame = subtracted_frame[edge_cut:-1*edge_cut ,edge_cut :-1*edge_cut ]

            # Find location of star
            max_indices = np.where(subtracted_frame == np.nanmax(subtracted_frame))

            # Create mask around star to replace with background image
            psfrem = psf_removal_mask((max_indices[1]+2, max_indices[0]+2), inner_rad,\
                                      outer_rad, array_shape[1], array_shape[0]) 

            # Compute offset between background and target image
            if chops[indices[k]] == "CHOP_A":
                offset = np.nanmedian(bg[:int(array_shape[0]/2), :] - img[:int(array_shape[0]/2), :])
            else:
                offset = np.nanmedian(bg[int(array_shape[0]/2):, :] - img[int(array_shape[0]/2):, :])

            # Replace star and load into array
            buf_3D[k] = img*(1-psfrem)+(bg-offset)*psfrem
    
            hdul.close()
            unsubtracted.close()
    
            count += 1
            
        stacked_img = np.nanmean(buf_3D, axis=0)*count
        
        return stacked_img, count

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
        bigarr, filecounts = zip(*tqdm(pool.imap(IntegrateFrames((array_shape)), filebufs),\
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
    file_size = float(hdul[0].data[0].nbytes)
    array_shape = np.shape(hdul[0].data[0])
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
        a_bigarr, a_filecounts = zip(*tqdm(pool.imap(IntegrateFrames((array_shape)), a_filebufs),\
                                           desc="integrating files", total=len(a_filebufs)))

    # Create chopa flat by mean of the images
    chopa_mean_frame = np.sum(a_bigarr, axis=0)/np.sum(a_filecounts)
    
    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        b_bigarr, b_filecounts = zip(*tqdm(pool.imap(IntegrateFrames((array_shape)), b_filebufs),\
                                           desc="integrating files", total=len(b_filebufs)))

    # Create chopb flat by mean of the images
    chopb_mean_frame = np.sum(b_bigarr, axis=0)/np.sum(b_filecounts)

    # Stitch chops together such that the halves of the image without the target are combined together
    if chop_direction == "UP-DOWN":

        flat = np.concatenate((chopb_mean_frame[:int(array_shape[0]/2), :], chopa_mean_frame[int(array_shape[0]/2):, :]))

    elif chop_direction == "LEFT-RIGHT":

        flat = np.concatenate((chopb_mean_frame[:, :int(array_shape[1]/2)].T, chopa_mean_frame[:, int(array_shape[1]/2):].T)).T

    elif chop_direction == "SEPARATE":

        # Remove zero values from the flat, replace with the minimum value    
        chopa_mean_frame[chopa_mean_frame == 0] = np.min(chopa_mean_frame[chopa_mean_frame != 0]) 
        chopb_mean_frame[chopb_mean_frame == 0] = np.min(chopb_mean_frame[chopb_mean_frame != 0]) 

        # Return chops separately
        return chopa_mean_frame, chopb_mean_frame        

    else:

        raise ValueError("Invalid chop direction")
        
    # Remove zero values from the flat, replace with the minimum value    
    flat[flat == 0] = np.min(flat[flat != 0]) 

    return flat

def create_proper_flat(files, chops, combined_flat, inner_rad = 27, outer_rad = 32, edge_cut=2, nbg=1,
                        tolerance=0.9, threadcount=50):
    
    """
    Creates a stacked flat frame for each chop state. They are either combined into a single flat or returned separately.
    
    Parameters:
    ----------------------
    files: list or array 
        List of raw file paths to stack.
    chops: list or array
        List of chop states corresponding to the file list, entries are either "CHOP_A" or "CHOP_B".
    combined_flat: 2D image array
        A combined flat for both chop states that is used solely to help identify tha target PSF location.
    inner_radius (optional): float
        Radius at which all pixels within the radius of the PSF removal mask are set to 1. Default value is 27 pixels.
    outer_radius (optional): float
        Radius at which all pixels outside the radius of the PSF removal mask are set to 0. Default value is 32 pixels.
    edge_cut (optional): integer
        Number of pixels to remove from the edges of the image before high pass filtering. Default value is 2 pixels.
    nbg (optional): integer
        Number of frames to use in rolling background subtraction
    tolerance (optional): float
        Fraction of available memory to be used for integration. Default value is 0.9.
    threadcount (optional): integer
        Number of threads to employ in multithreading. Default value is 50 threads.
    
    Returns: 
    ---------------------- 
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
    chopa_files = np.where(chops == "CHOP_A")[0]
    chopb_files = np.where(chops == "CHOP_B")[0]

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
        a_bigarr, a_filecounts = zip(*tqdm(pool.imap(IntegrateFrames_Flat((files, chops, combined_flat, nbg, array_shape, edge_cut, inner_rad, outer_rad)), a_filebufs),\
                                           desc="integrating files", total=len(a_filebufs)))

    # Create chopa flat by mean of the images
    chopa_mean_frame = np.sum(a_bigarr, axis=0)/np.sum(a_filecounts)
    
    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        b_bigarr, b_filecounts = zip(*tqdm(pool.imap(IntegrateFrames_Flat((files, chops, combined_flat, nbg, array_shape, edge_cut, inner_rad, outer_rad)), b_filebufs),\
                                           desc="integrating files", total=len(b_filebufs)))

    # Create chopb flat by mean of the images
    chopb_mean_frame = np.sum(b_bigarr, axis=0)/np.sum(b_filecounts)

    # Remove zero values from the flat, replace with the minimum value 
    chopa_mean_frame[chopa_mean_frame == 0] = np.min(chopa_mean_frame[chopa_mean_frame != 0]) 
    chopb_mean_frame[chopb_mean_frame == 0] = np.min(chopb_mean_frame[chopb_mean_frame != 0]) 
     
    return chopa_mean_frame, chopb_mean_frame
