#----------------------------------------
# IMPORTS
#----------------------------------------
import os, pathlib
import numpy as np
from tqdm.auto import tqdm
from multiprocessing.pool import ThreadPool as Pool

from astropy.io import fits

from scipy.signal import correlate

import pyNOMIC.helper_functions as hf

#----------------------------------------
# CLASSES
#----------------------------------------

class BinFrames(object):

    '''
    Bin a list of frames and return the frame or save it as a file.
    '''
    
    def __init__(self, params):

        """
        Parameters (contained inside a tuple):
        ----------------------
        array_shape: integer tuple
            Tuple containing image dimensions, from numpy.shape
        binned_dir: string/Path object
            Path to save binned image, if value is None the image is
            returned instead of saved
        """
        
        self.params = params
        self.integrator = hf.IntegrateFrames(params[0])
    
    def __call__(self, angles_files_tuple):

        """
        Parameters:
        ----------------------
        angles_files_tuple: tuple containing 'angles' and 'files'

            angles: list or array
                List of parallactic angles corresponding to each file
                in 'files'
            files: list or array 
                List of raw file paths, sorted 
             
        Returns: 
        ---------------------- 
        binned frame: 2D image array
            binned frame (averaged), only returned if binned_dir is None
        mean_parallactic_angle: float
            Average parallactic angle over the frames
        filename: string
            The filename of the first file in the bin, with the prefix
            "binned_" added on
        """

        array_shape, binned_dir = self.params
        angles, files = angles_files_tuple

        # Integrate frames to get binned frame
        binned_frame, count = self.integrator(files)

        # IntegrateFrames multiplies the averaged frames by the count, need to divide it out
        binned_frame = binned_frame/count

        filename = "binned_"+files[0].name

        # Find the average parallactic angle over the bin
        mean_parallactic_angle = np.nanmean(angles)

        if binned_dir is None:
            
            return binned_frame, mean_parallactic_angle, filename
            
        else:

            # Write image to path
            newhdul = fits.HDUList([fits.PrimaryHDU(data=binned_frame/count)])
            newhdul.writeto(os.path.join(binned_dir, filename), overwrite=True)
            newhdul.close()
    
            return mean_parallactic_angle, filename

class EvaluateFrames(object):
    
    def __init__(self, params):

        """
        Parameters (contained inside a tuple):
        ----------------------
        img_files: string/Path object
            Directory where images are read from
        chopa_integrated: 2D image array
            Stacked image frame for CHOP_A
        chopb_integrated: 2D image array
            Stacked image frame for CHOP_B
        wx: 1D numpy array
            Array enumerating columns of the alignment grid
        wy: 1D numpy array
            Array enumerating rows of the alignment grid
        windowsize: integer
            Half width/height of the reference cutout image
            (which is 1:1 aspect ratio)
        array_shape: integer tuple
            Tuple containing image dimensions, from numpy.shape
        wvl_interp: 1D numpy array
            An array of wavelengths. 
        relative_flux: 1D numpy array
            Relative flux for each respective wavelength in wvl_interp.
        model_trefoil: boolean
            Enables the modeling of trefoil in the PSF.
            Enabled by default.
        subtract_psf: boolean
            Enables saving psf subtracted images. Disabled by default.
        psf_subtracted_dir: string
            Directory in which to save PSF subtracted images.
        """
        
        self.params = params
        
    def __call__(self, chop_file_tuple):

        """
        Parameters:
        ----------------------
        chop_file_tuple: tuple containing 'chop' and 'file'

            chop: string
                Current chop states, either "CHOP_A" or "CHOP_B"
                
            file: string/Path object
                Path to raw image file
        Returns:
        ----------------------    
        psfmaxima: float
            Measured maximum pixel value of the PSF.
        background_dev: float
            Measured standard deviation of the background.
        corr: float
            Maximum value of the cross correlation of the frame and
            its respective mean frame.
        std_residual: float
            Standard deviation of the residual of PSF subtraction.  
        lbtfit: 1D numpy array
            Empirical psf fitting parameters.
        reffit: 1D numpy array
            Airy disk fitting parameters.
        image: 2D numpy array
            PSF subtracted image. Is None if PSF subtraction
            is disabled.
        """
        
        (img_files, chopa_integrated, chopb_integrated, wx, wy, windowsize, array_shape,
         wvl_interp, relative_flux, model_trefoil, subtract_psf, psf_subtracted_dir) = self.params

        chop, file = chop_file_tuple

        # if an array of files is not provided, file is a path
        if img_files is None:
            hdul = fits.open(file)
            frame = hdul[0].data
            hdul.close()
            
        # if an array of files is provided, file is an index
        else:
            frame = img_files[int(file)]
        if subtract_psf:
            image = np.copy(frame)
        else:
            image = None

        # Create mask to mask out star
        max_mask =  hf.circular_mask(((array_shape[1]/2 - 0.5), (array_shape[0]/2 - 0.5)), 
                                     windowsize, array_shape[0], array_shape[1])
        
        # Compute background deviation by excluding values 3 sigma above the image median
        background_dev = np.nanstd(frame[~max_mask])

        # Remove all nans for cross correlation, replace with 0s
        frame[np.isnan(frame)] = 0

        # Use corresponding average frame based on chop state
        if chop == "CHOP_B":
            corr = (np.max(correlate(chopb_integrated, frame)))
        else:
            corr = (np.max(correlate(chopa_integrated, frame)))    

        # Create image cutout for airy fitting
        cutout = frame[(int(array_shape[0]/2)-windowsize):(int(array_shape[0]/2)+windowsize),
                       (int(array_shape[1]/2)-windowsize):(int(array_shape[1]/2)+windowsize)]
        
        # Get the value of the maximum
        psfmaxima = np.nanmax(cutout)
        
        try:

            # Fit cutout to get empirical psf parameters
            reffit, lbtfit, trifit = hf.empirical_psf_fit(cutout, wvl_interp, relative_flux,
                                                          model_trefoil=model_trefoil)

            # Create psf model by integrating over wavelength
            psf_model = lbtfit[0]*np.mean(hf.modified_airy_disk((wx, wy), relative_flux,
                                                                wvl_interp, lbtfit[2], lbtfit[3],
                                                                0, reffit[4], reffit[5],
                                                                reffit[6]),
                                          axis=0)/np.mean(relative_flux) + lbtfit[1]

            if model_trefoil:
                # Add trefoil model
                psf_model += hf.center_triangle((wx, wy), trifit[0], trifit[1], trifit[2],
                                                trifit[3], trifit[4], reffit[5], reffit[6],
                                                trifit[5],trifit[6], ravel=False)
            # Subtract psf_model from cutout
            residual = cutout - psf_model

            # To get PSF subtracted, create model with the entire image
            if subtract_psf:

                origin = (int(array_shape[1]/2)-windowsize+reffit[5],
                          int(array_shape[0]/2)-windowsize+reffit[6])

                # Create model grid of the entire image
                nx = np.linspace(0, array_shape[1]-1, array_shape[1])
                ny = np.linspace(0, array_shape[0]-1, array_shape[0])
                nx, ny = np.meshgrid(nx, ny)

                
                # Create psf model by integrating over wavelength
                psf_model = lbtfit[0]*np.mean(hf.modified_airy_disk((nx, ny), relative_flux,
                                                                    wvl_interp, lbtfit[2],
                                                                    lbtfit[3], 0, reffit[4],
                                                                    origin[0], origin[1]),
                                              axis=0)/np.mean(relative_flux) + lbtfit[1]
    
                if model_trefoil:
                    # Add trefoil model
                    psf_model += hf.center_triangle((nx, ny), trifit[0], trifit[1], trifit[2],
                                                    trifit[3], trifit[4], origin[0], origin[1],
                                                    trifit[5],trifit[6], ravel=False)

                # Subtract psf
                image = image - psf_model

                if psf_subtracted_dir is not None:
                    # Write image to file
                    newhdul = fits.HDUList([fits.PrimaryHDU(data=(image))])
                    newhdul.writeto(os.path.join(psf_subtracted_dir,
                                                 "psfsubtracted_"+files[i].name), overwrite=True)
                    newhdul.close()

                    image = None

        except:

            image = None
            
            return (psfmaxima, background_dev, corr, np.nan, lbtfit, reffit, image)
            
        return (psfmaxima, background_dev, corr, np.nanstd(residual), lbtfit, reffit, image)

#----------------------------------------
# FUNCTIONS
#----------------------------------------

def frame_evaluation(aligned_files, chops, array_shape, file_size, stellar_temp, pxscale=0.0179,
                     windowsize=20, model_trefoil=True, subtract_psf=False,
                     psf_subtracted_dir=None, method="median", buffer_type="frames",
                     tolerance=0.9, memoryMode=1, threadcount=50):

    """
    Evaluates all the frames, measuring the FWHM, eccentricities,
    maxima, background deviations, maximum cross correlation,
    and empirical PSF fit parameters.
    
    Parameters:
    ----------------------
    aligned_files: list or array
        List of aligned file paths, sorted 
    chops: string array
        List of chop states corresponding to the file list,
        entries are either "CHOP_A" or "CHOP_B"
    array_shape: integer tuple
        Tuple containing image dimensions, from numpy.shape
    file_size: float
        File size of aligned images.
    stellar_temp: float
        Temperature of the star in Kelvins.
    pxscale (optional): float
        The image scale of the images in arcseconds per pixel.
        Default is 0.0179"/px, for LBTI-NOMIC.
    windowsize (optional): integer
        Half width/height of the reference cutout image
        (which is 1:1 aspect ratio). Default is 20 pixels.
    subtract_psf (optional): boolean
        Enables saving psf subtracted images. Disabled by default.
    psf_subtracted_dir (optional): string
        Directory in which to save PSF subtracted images.
    method (optional): string
        Method to  create a stacked image, either by taking the "mean"
        or "median". Default is "median".
    buffer_type (optional): string
        Specifies which type of buffer to use, either "files" or
        "frames". Set to "frames" by default. Note that using median
        integration and file buffers are incompatible.
    tolerance (optional): float
        Fraction of available memory to be used for integration.
    memoryMode (optional): integer
        If set to 0, frames are retrieved from aligned_files as an
        image cube (from memory). If set to 1, frames are opened from
        aligned_files as a list of files. Default is 1.
    threadcount (optional): integer
        Number of threads to employ in multithreading.
        Default value is 50 threads.

    Returns:
    ----------------------     
    fwhms: 1D numpy array
        The measured FWHMs of the PSF, from the airy_disk model
        standard deviations.
    eccentricities: 1D numpy array
        The measured eccentricities of the PSF, from the airy_disk
        model standard deviations.
    psfmaxima: 1D numpy array
        Measured maximum pixel values of the PSF.
    background_dev: 1D numpy array
        Measured standard deviations of the background.
    corr: 1D numpy array
        Maximum values of the cross correlation of the frames
        and their respective mean frames.
    std_residuals: 1D numpy array
        Standard deviations of the residual of PSF subtraction.
    lbtfits: 2D numpy array
        Empirical psf fitting parameters (floats)
    reffits: 2D numpy array
        Airy disk fitting parameters (floats)
    images: 3D numpy array
        PSF subtracted images. Is None if PSF subtraction
        is disabled or if the images are saved.
    """

    wx = np.linspace(0, 2*windowsize-1, 2*windowsize)
    wy = np.linspace(0, 2*windowsize-1, 2*windowsize)
    wx, wy = np.meshgrid(wx, wy)

    wvl_interp, relative_flux = hf.calculate_expected_flux(stellar_temp)

    # Files are in memory
    if memoryMode == 0:

        # Create averaged frames for each chop state
        if method == "mean":
            chopa_integrated = np.nanmean(aligned_files[chops == "CHOP_A"], axis=0)
            chopb_integrated = np.nanmean(aligned_files[chops == "CHOP_B"], axis=0)
        else:
            chopa_integrated = np.nanmedian(aligned_files[chops == "CHOP_A"], axis=0)
            chopb_integrated = np.nanmedian(aligned_files[chops == "CHOP_B"], axis=0)
            
        # Set nans to 0 allow correlation to proceed
        chopb_integrated[np.isnan(chopb_integrated)] = 0
        chopa_integrated[np.isnan(chopa_integrated)] = 0

        #if __name__ == "__main__":
        with Pool(threadcount) as pool:
            (psfmaxima, background_dev, correlations,
             residual_dev, lbtfits, reffits, images) =\
             zip(*tqdm(pool.imap(EvaluateFrames((aligned_files, chopa_integrated, chopb_integrated,
                                                 wx, wy, windowsize, array_shape, wvl_interp,
                                                 relative_flux, model_trefoil, subtract_psf,
                                                 psf_subtracted_dir)),
                                 np.array((chops, np.arange(len(aligned_files)))).T),
                       total=len(aligned_files), desc="Evaluating frames"))
        
    else:
    
        if buffer_type == "frames":
            chopa_integrated = hf.integrate_frames_buffer(aligned_files[chops=="CHOP_A"],
                                                          method=method,
                                                          tolerance=tolerance,
                                                          threadcount=threadcount)
            chopb_integrated = hf.integrate_frames_buffer(aligned_files[chops=="CHOP_B"],
                                                          method=method,
                                                          tolerance=tolerance,
                                                          threadcount=threadcount)
        else:        
            if method == "mean":
                chopa_integrated = hf.integrate_files_buffer(aligned_files[chops=="CHOP_A"],
                                                             tolerance=tolerance,
                                                             threadcount=threadcount)
                chopb_integrated = hf.integrate_files_buffer(aligned_files[chops=="CHOP_B"],
                                                             tolerance=tolerance,
                                                             threadcount=threadcount)
            else:
                raise ValueError("Incompatible integration method and buffer type")

        chopb_integrated[np.isnan(chopb_integrated)] = 0
        chopa_integrated[np.isnan(chopa_integrated)] = 0

        #if __name__ == "__main__":
        with Pool(threadcount) as pool:
            (psfmaxima, background_dev, correlations,
             residual_dev, lbtfits, reffits, images) =\
             zip(*tqdm(pool.imap(EvaluateFrames((None, chopa_integrated, chopb_integrated, wx, wy,
                                                windowsize, array_shape, wvl_interp, relative_flux,
                                                 model_trefoil, subtract_psf, psf_subtracted_dir)),
                                 np.array((chops, aligned_files)).T), total=len(aligned_files),
                       desc="Evaluating frames"))

    # Convert lists into numpy arrays
    lbtfits, reffits = np.asarray(lbtfits), np.asarray(reffits)

    # Calculate fwhms
    fwhms = 2*np.sqrt(2*np.log(2)*(reffits[:,1]+reffits[:,2]))*pxscale

    # Calculate eccentricities, handling the potential imaginary radical
    eccentricities = np.sqrt(1 - reffits[:,1]/reffits[:,2])
    eccentricities_r = np.sqrt(1 - reffits[:,2]/reffits[:,1])
    eccentricities[np.isnan(eccentricities)] = eccentricities_r[np.isnan(eccentricities)]
    
    return (fwhms, eccentricities, np.asarray(psfmaxima), np.asarray(background_dev),
            np.asarray(correlations), np.asarray(residual_dev), lbtfits, reffits, images)

def frame_rejection(chops, params, sigma=None):

    """
    Rejects frames based on given parameters and standard deviation
    thresholds.
    
    Parameters:
    ----------------------
    chops: string array
        List of chop states corresponding to the file list,
        entries are either "CHOP_A" or "CHOP_B"
    params: list or numpy array
        List containing parameter arrays corresponding to the file list
    sigma (optional): list or numpy array
        List containing standard deviation thresholds corresponding to
        each parameter in params. Default is 1.5 sigma for
        each parameter
        
    Returns: 
    ---------------------- 
    bools: boolean array
        Boolean mask array denoting which frames are rejected
        (rejected frame indices are set to False)
    """
    
    # Find chop states
    chopa_bool = (chops == "CHOP_A")
    chopb_bool = (chops == "CHOP_B")

    # if sigma is None, use sigma=1.5 for all parameters
    if sigma is None:
        sigma = 1.5*np.ones(len(params))

    # Compensate for systematic differences between chop states by transforming parameters
    for i in range(len(params)):
        
        params[i][chopa_bool] *= np.nanstd(params[i][chopb_bool])/np.nanstd(params[i][chopa_bool])
        params[i][chopa_bool] += np.nanmedian(params[i][chopb_bool]) -\
                                 np.nanmedian(params[i][chopa_bool])

    # Initialize boolean array by removing nans in first parameter
    bools = ~np.isnan(params[0])

    # Iterate over parameters to reject frames
    for i in range(len(params)):
        
        bools = bools & (params[i] < np.nanmedian(params[i]) + sigma[i]*np.nanstd(params[i]))

    return bools
    
'''
def fractional_frame_rejection(psfmaxima, background_dev, fwhms, eccentricities, correlations,
                               amplitudes, offsets,fraction_frames=0.3, start_sigma=5, fev=100):

    sigma = start_sigma+0.1
    frac_frame_bool = 1
    count = 0

    while (count < fev):
        sigma -= 0.1
        frame_bool = frame_rejection(psfmaxima, background_dev, fwhms, eccentricities,
                                     correlations, amplitudes, offsets, sigma=sigma)
        frac_frame_bool = len(frame_bool[frame_bool == False])/len(frame_bool)
        if (frac_frame_bool >= fraction_frames):
            return frame_bool, sigma
        count += 1
    raise ValueError("Exceeded number of iterations!")
'''

def frame_binning(aligned_files, chops, para_angles, frame_bool, array_shape, bin=50, prefix='',
                  memoryMode=1, threadcount=50):

    """
    Bin frames temporally, and update chop states and parallactic angle.
    
    Parameters:
    ----------------------
    aligned_files: list or array 
        List of aligned file paths, sorted.
    chops: string array
        List of chop states corresponding to the file list,
        entries are either "CHOP_A" or "CHOP_B"
    para_angles: list or array
        List of parallactic angles corresponding to each file in
        'aligned_files'
    frame_bool: boolean array
        Boolean mask array denoting which frames to exclude
        from binning.
    array_shape: integer tuple
        Tuple containing image dimensions, from numpy.shape
    bin (optional): integer
        Number of frames to add for each binned frame.
        Default value is 50 frames.
    prefix (optional): string
        Prefix to add to directory name when saving image.
    memoryMode (optional): integer
        If set to 0, frames are returned to memory.
        If set to 1, frames are saved to files.
        Default is 1.
    threadcount (optional): integer
        Number of threads to employ in multithreading.
        Default value is 50 threads.
        
    Returns: 
    ---------------------- 
    binned_files: Path array or array of 2D images
        If memoryMode is 0, this variable contains all of the binned
        frames as an image cube. Otherwise, this variable is a array of
        file paths to the binned frames.
    binned_chops: string array
        List of chop states corresponding to each image in
        'binned_files', entries are either "CHOP_A" or "CHOP_B"
    binned_angles: array
        List of parallactic angles corresponding to each image
        in 'binned_files'
    """

    # Create binned directory
    root_dir = os.path.dirname(os.path.dirname(aligned_files[0]))
    binned_dir=os.path.join(root_dir,prefix+'binned')

    if not os.path.exists(binned_dir):
        os.makedirs(binned_dir)

    # Separate files and angles by chop state
    chopa_files = aligned_files[frame_bool & (chops == "CHOP_A")]
    chopb_files = aligned_files[frame_bool & (chops == "CHOP_B")]
    chopa_angles = para_angles[frame_bool & (chops == "CHOP_A")]
    chopb_angles = para_angles[frame_bool & (chops == "CHOP_B")]

    # Calculate buffer size from bin size (memory is assumed to be available)
    a_buffer = int(np.ceil(len(chopa_files)/bin))
    b_buffer = int(np.ceil(len(chopb_files)/bin))
    
    # Split files and angles into buffers
    a_splitlist = np.linspace(0, len(chopa_files), 1+a_buffer)[1:-1].round().astype(int)
    a_binfiles = np.split(chopa_files, a_splitlist)
    a_angles = np.split(chopa_angles, a_splitlist)
    b_splitlist = np.linspace(0, len(chopb_files), 1+b_buffer)[1:-1].round().astype(int)
    b_binfiles = np.split(chopb_files, b_splitlist)
    b_angles = np.split(chopb_angles, b_splitlist)

    print("Binning files...")

    # Calculate number of bin files that will be created
    numbinfiles = len(a_binfiles) + len(b_binfiles)

    # Create arrays
    binned_chops = np.empty(numbinfiles, dtype="<U16")
    binned_angles = np.zeros(numbinfiles)

    # Return frames in memory
    if memoryMode == 0:

        binned_frames = np.zeros((numbinfiles, array_shape[0], array_shape[1]))
        
        #if __name__ == "__main__":
        with Pool(threadcount) as pool:
            a_binned_frames, a_binned_angles, a_binned_filenames =\
            zip(*tqdm(pool.imap(BinFrames((array_shape, None)), zip(a_angles, a_binfiles))))

        #if __name__ == "__main__":
        with Pool(threadcount) as pool:
            b_binned_frames, b_binned_angles, b_binned_filenames =\
            zip(*tqdm(pool.imap(BinFrames((array_shape, None)), zip(b_angles, b_binfiles))))

        # Sort filenames in order
        binned_filenames = np.asarray(sorted(a_binned_filenames+b_binned_filenames))

        # Convert into numpy arrays
        a_binned_frames, a_binned_angles, a_binned_filenames = (np.asarray(a_binned_frames),
                                                                np.asarray(a_binned_angles),
                                                                np.asarray(a_binned_filenames))
        b_binned_frames, b_binned_angles, b_binned_filenames = (np.asarray(b_binned_frames),
                                                                np.asarray(b_binned_angles),
                                                                np.asarray(b_binned_filenames))

        # Populate chop list with chop states
        binned_chops[np.where(np.isin(binned_filenames,a_binned_filenames) == True)[0]] = "CHOP_A"
        binned_chops[np.where(np.isin(binned_filenames,b_binned_filenames) == True)[0]] = "CHOP_B"

        # Populate binned_angles and binned_frames with angles and frames in order
        for i in range(numbinfiles):

            if binned_filenames[i] in a_binned_filenames:
                
                binned_angles[i] = a_binned_angles[np.where(a_binned_filenames\
                                                            == binned_filenames[i])[0]][0]
                binned_frames[i] = a_binned_frames[np.where(a_binned_filenames\
                                                            == binned_filenames[i])[0]][0]
                
            else:
                
                binned_angles[i] = b_binned_angles[np.where(b_binned_filenames\
                                                            == binned_filenames[i])[0]][0]
                binned_frames[i] = b_binned_frames[np.where(b_binned_filenames\
                                                            == binned_filenames[i])[0]][0]
        
        return binned_frames, binned_chops, binned_angles

    # Save frames to files and return paths
    else:
            
        #if __name__ == "__main__":
        with Pool(threadcount) as pool:
            a_binned_angles, a_binned_filenames = zip(*tqdm(pool.imap(BinFrames((array_shape,
                                                                                 binned_dir)),
                                                                      zip(a_angles, a_binfiles))))
            
        a_binned_angles, a_binned_filenames = (np.asarray(a_binned_angles),
                                               np.asarray(a_binned_filenames))
        
        #if __name__ == "__main__":
        with Pool(threadcount) as pool:
            b_binned_angles, b_binned_filenames = zip(*tqdm(pool.imap(BinFrames((array_shape,
                                                                                 binned_dir)),
                                                                      zip(b_angles, b_binfiles))))
            
        b_binned_angles, b_binned_filenames = (np.asarray(b_binned_angles), 
                                               np.asarray(b_binned_filenames))

        # Create file list by reading path
        binned_files = sorted(list(pathlib.Path(str(binned_dir)).rglob('*.fits')))
        binned_files = np.asarray([a for a in binned_files if a.name[0]!='.'\
                                   and str(a.parent)==binned_dir])

        # Get filenames
        binned_filenames = np.asarray([f.name for f in binned_files])

        # Populate chop list with chop states
        binned_chops[np.where(np.isin(binned_filenames,a_binned_filenames) == True)[0]] = "CHOP_A"
        binned_chops[np.where(np.isin(binned_filenames,b_binned_filenames) == True)[0]] = "CHOP_B"

        # Populate binned_angles and binned_frames with angles and frames in order
        for i in range(len(binned_files)):
            
            if binned_filenames[i] in a_binned_filenames:
                
                binned_angles[i] = a_binned_angles[np.where(a_binned_filenames\
                                                            == binned_filenames[i])[0]][0]
                
            else:
                
                binned_angles[i] = b_binned_angles[np.where(b_binned_filenames\
                                                            == binned_filenames[i])[0]][0]
        
        return binned_files, binned_chops, binned_angles
