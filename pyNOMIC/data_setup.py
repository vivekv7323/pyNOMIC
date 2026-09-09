#----------------------------------------
# IMPORTS
#----------------------------------------
import os, pathlib
import numpy as np
from itertools import groupby
from tqdm.auto import tqdm
from multiprocessing.pool import ThreadPool as Pool

from astropy.io import fits
from astropy.time import Time
from astropy.convolution import (convolve_fft, Box2DKernel,
                                 Ring2DKernel, Gaussian2DKernel)

from scipy.ndimage import maximum_filter1d

import pyNOMIC.helper_functions as hf

#----------------------------------------
# CLASSES
#----------------------------------------

class FileInfo(object):

    '''
    Open raw images and obtain information from fits header.
    '''

    def __init__(self, params):

        """
        Parameters (contained inside a tuple):
        ----------------------
        new_raw_dirs: list
            List containing two directory paths for spliting
            the raw data in the case of double-sided imaging.
            Is set to None in the case of single-sided imaging.
        obj: string
            Object name, should match the image header
        skip_target_check: boolean
            if True, skips checking the image header
            for the target name 'obj'
        recalc_para_angles: boolean
            If True, recalculates parallactic angle
        frame_median_limit: integer
            Limit for the median of the frame. Frames with median above
            this limit are rejected. Default is 28000 counts.
        cold_stop_crop: integer
            Number of pixels to mask along the cold stop in
            the case of double-sided imaging.
        correct_linearity: function
            Linearity correction function.
            Is None if linearity correction is disabled.
        """
        
        self.params = params
        
    def __call__(self, file):
        
        """
        Parameters:
        ----------------------
        file: string/Path object
            Path to raw image file

        Returns:
        ----------------------    
        chop: string
            The chop state identified from the image header
        frame_median: integer
            The median value of the frame
        para_angle: float
            The parallactic angle identified from
            the image header in degrees.
        end_time: float
            End time of exposure (Julian date).
        temp: float
            Air temperature in degrees Celsius.
        airmass: float
            Airmass
        wind_spd: float
            Wind speed in meters per second.
        wind_dir: float
            Wind direction, degrees East of North.
        seeing: float
            DIMM seeing in arcseconds.
        pwv: float
            SMT Precip water vapor, 0.05*mm
        exp_time: float
            Nominal total integration time per pixel.
        ncoadds: integer
            Number of coadded frames in image.
        channel_medians: 2D numpy array (8 X N)
            Medians of each of the 8 channels in each image.     
        channel_stds: 2D numpy array (8 X N)
            Standard deviations of each of the
            8 channels in each image.
        """
        
        (new_raw_dirs, obj, skip_target_check, recalc_para_angles,
         frame_median_limit, cold_stop_crop, correct_linearity) = self.params

        hdul = fits.open(file)
        orig = hdul[0].data[0]

        if correct_linearity is not None:
            orig = correct_linearity(orig)
            if len(new_raw_dirs) != 2:

                newhdul = fits.HDUList([fits.PrimaryHDU(data=orig)])
                newhdul.writeto(os.path.join(new_raw_dirs[0], file.name),
                                overwrite=True)
                newhdul.close()               

        # Check if correct object
        if not skip_target_check:
            try:
                object = hdul[0].header['OBJNAME']
            except:
                print(file)
                raise ValueError("OBJNAME header not found")
            if object != obj:
                raise ValueError("Object in header does not match given object")

        # Get frame median
        frame_median = np.median(orig)
        channel_medians, channel_stds = hf.channel_stats(orig)

        # Only process if frame median is low enough
        if frame_median < frame_median_limit:
            
            # Obtain chop state if available
            try:
                chop = hdul[0].header['CHOP_POS']
            except:
                chop = "CHOP_NA"

            # Obtain parallactic angle
            if recalc_para_angles:
                try:
                    para_angle = hf.calc_para_angles(hdul[0].header['LBT_LST'],
                                                     hdul[0].header['LBT_RA'],
                                                     hdul[0].header['LBT_DEC'])
                except:
                    para_angle = hf.para_angle_query(hdul[0].header["DATE-OBS"],
                                                     hdul[0].header["TIME-OBS"],
                                                     hdul[0].header["OBJNAME"])
            else:
                para_angle = float(hdul[0].header['LBT_PARA'])   

            # Obtain other fits header information
            time = Time(hdul[0].header['DATE-OBS'] +"T"+ hdul[0].header['TIME-END'],
                        format='isot', scale='utc')
            
            end_time = float(time.jd)
            try:
                temp = float(hdul[0].header['LBTTEMP'])
            except:
                temp = np.nan
            try:
                airmass = float(hdul[0].header['LBT_AIRM'])
            except:
                airmass = np.nan
            try:
                wind_spd = float(hdul[0].header['WINDSPD'])
            except:
                wind_spd = np.nan
            try:
                wind_dir = float(hdul[0].header['WINDDIR'])
            except:
                wind_dir = np.nan
            try:
                seeing = float(hdul[0].header['SEEING'])
            except:
                seeing = np.nan
            try:
                # SMT Precip water vapor
                pwv = float(hdul[0].header['SMTTAU'])
            except:
                pwv = np.nan
            exp_time = float(hdul[0].header['EXPTIME'])
            ncoadds = int(hdul[0].header['NCOADDS'])
            #nod_pos, dettemp, nomiccfw not found
            
            # Do if data is double sided
            if len(new_raw_dirs) == 2:

                array_shape = np.shape(hdul[0].data[0])
    
                # Write split raw images to path
                newhdul = fits.HDUList([fits.PrimaryHDU(data=\
                                            [orig[:int(0.5*array_shape[0]-cold_stop_crop), :]])])
                newhdul.writeto(os.path.join(new_raw_dirs[0], file.name[:-5]+"_sx.fits"),
                                overwrite=True)
                newhdul.close()
    
                newhdul = fits.HDUList([fits.PrimaryHDU(data=\
                                            [orig[cold_stop_crop+int(0.5*array_shape[0]):, :]])])
                newhdul.writeto(os.path.join(new_raw_dirs[1], file.name[:-5]+"_dx.fits"),
                                overwrite=True)
                newhdul.close()

            hdul.close()

            return (chop, frame_median, para_angle, end_time, temp, airmass,
                    wind_spd, wind_dir, seeing, pwv, exp_time, ncoadds,
                    channel_medians, channel_stds)
            
        else:

            hdul.close()

            return ("CHOP_NA", np.nan, np.nan, np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                    np.nan, np.nan)
            
class HighPass(object):

    '''
    Open raw images and create high pass filtered images.
    '''

    def __init__(self, params):

        """
        Parameters (contained inside a tuple):
        ----------------------
        highpass_dir: string/Path object
            Path to save high pass filtered images
        bools: 2D Boolean array
            Bad pixel map image converted into a boolean array.
        highpassmask: 2D boolean array
            mask applied to images before high pass filtering,
            affected pixels are set to np.nan
        tempflat: 2D numpy array
            Temporary flat applied to locate the star
        smooth: integer
            Outer radius of the star mask
        binsize: integer
            Number of pixels for spatial binning.
        """

        self.params = params

    def __call__(self, file):

        """
        Parameters:
        ----------------------
        file: string/Path object
            Path to raw image file
        """

        (highpass_dir, bools, highpassmask,
         tempflat, smooth, binsize) = self.params

        hdul = fits.open(file)

        orig = hdul[0].data[0]

        # If flat is available, create flat divided image
        if tempflat is not None:
            image = orig/tempflat
        else:
            image = np.copy(orig)

        # image buffer required for a certain value of the convolution kernel radius
        smooth_buf = int(1.25*smooth)

        # Replace bad pixels or nan regions with the median for convolution
        image[~bools] = np.nanmedian(image)

        # Use highpass filter mask if available
        if highpassmask is not None:
            image[~highpassmask] = np.nan

        # Create copy of image to convolve
        new_image = np.copy(image)

        # Pad image to prevent the box kernel from introducing an edge gradient
        new_image = np.pad(new_image, smooth_buf, mode='edge')

        # Subtract convolved image from original image
        filtered_frame = convolve_fft(image -
                                      convolve_fft(new_image, Box2DKernel(smooth))\
                                          [smooth_buf:-1*smooth_buf, smooth_buf:-1*smooth_buf],
                                      Box2DKernel(binsize*3))

        filtered_frame,_ = hf.spatial_binning([filtered_frame], binsize)   
                
        # Write high pass image to path
        newhdul = fits.HDUList([fits.PrimaryHDU(data=filtered_frame[0])])
        newhdul.writeto(os.path.join(highpass_dir, "highpass_"+file.name), overwrite=True)
        newhdul.close()
                
        hdul.close()

        return True, True
    
class ChopMetrics(object):

    """
    Attempt at parallelized/non-stellar chop detection.
    """

    def __init__(self, params):
        
        """
        Parameters (contained inside a tuple):
        ----------------------
        files: list or array 
            List of raw file paths, sorted 
        directory: string/Path object
            Path to files to perform chop detection
        array_shape: integer tuple
            Tuple containing image dimensions, from numpy.shape
        chop_direction: string
            The chopping direction employed, either "UP-DOWN", "LEFT-RIGHT",
            or "SEPARATE" if separate flats for each chop are desired.
            Default value is "UP-DOWN", used for single-sided imaging.
        """
        
        self.params = params
        
    def __call__(self, i):

        """
        Parameters:
        ----------------------
        index: integer
            File index to process from 'files'

        Returns: 
        ---------------------- 
        minmax: float
            Difference between the minimum and maximum of the frame
            after chop subtraction.
        stdev: float
            Standard deviation of the frame after chop subtraction.
        bgmean: float
            Mean of the absolute value of the frame after
            chop subtraction.
        locdiff: float
            Difference in position vector between
            the maximum and minimum of the frame.
        maximum: float
            The maximum of the frame.
        chop_guess: string
            The initial guess of the chop state.
        index: integer
        """    

        files, directory, array_shape, chop_direction = self.params
        
        image = fits.open(os.path.join(directory, files[i].name))

        image2 = fits.open(os.path.join(directory, files[i+1].name))

        subtract =  convolve_fft(image[0].data[0] - image2[0].data[0],  Box2DKernel(10))
                
        image.close()
        image2.close()

        minmax = np.max(subtract) - np.min(subtract)
        stdev = np.std(subtract)
        bgmean = np.mean(np.abs(subtract))
        
        indices = np.where(subtract == np.nanmax(subtract))
        
        if (chop_direction == "UP-DOWN") or (chop_direction == "DIAGONAL"):
            
            locdiff = (np.where(subtract == np.max(subtract))[0][0] -
                       np.where(subtract == np.min(subtract))[0][0])

            if indices[0][0] < array_shape[0]/2:
                chop_guess = "CHOP_A"
            else:
                chop_guess = "CHOP_B"
                
        elif (chop_direction == "LEFT-RIGHT"):
            
            locdiff = (np.where(subtract == np.max(subtract))[1][0] -
                       np.where(subtract == np.min(subtract))[1][0])

            if indices[1][0] < array_shape[1]/2:
                chop_guess = "CHOP_A"
            else:
                chop_guess = "CHOP_B"     
            
        maximum = [indices[0][0], indices[1][0]]

        return minmax, stdev, bgmean, locdiff, maximum, chop_guess, i

class FourierMean(object):
    
    '''
    Calculate the mean power of the chop residual component
    of the Fourier spectrum of the image.
    '''

    def __init__(self, params):

        self.params = params

    def __call__(self, i):

        """
        Parameters:
        ----------------------
        index: integer
            File index to process from 'files'

        Returns: 
        ---------------------- 
        measure: float
            Mean power of the chop residual component
            of the Fourier spectrum of the image
        index: integer
        """

        files, sigma, cutout = self.params

        hdul = fits.open(files[i])
        if len(cutout) != 4:
            img = hdul[0].data[0]
        else:
            img = hdul[0].data[0][cutout[0]:cutout[1],
                                  cutout[2]:cutout[3]]
        hdul.close()
    
        # Compute 2D Fast Fourier Transform
        f_transform = np.fft.fft2(img)
        
        # Shift zero frequency component to the center
        f_shift = np.fft.fftshift(f_transform)
        
        # Calculate magnitude spectrum
        img2 = np.log(np.abs(f_shift))

        measure = np.mean(img2[(img2 < (np.median(img2) + sigma*np.std(img2)))])

        return measure, i

#----------------------------------------
# FUNCTIONS
#----------------------------------------

def create_badmap(files, sigma=1, smooth=30, edge_cut=3, growth=1,
                  buffer_type="files", tolerance=0.9, threadcount=50):

    """
    Create a badmap by constructing a crude flat, running it through a
    high pass filter, and masking out large deviations.
    
    Parameters:
    ----------------------
    files: list or array 
        List of raw file paths to stack.
    sigma (optional): float
        Number of standard deviations to include in the badmap.
    smooth (optional): integer
        Outer radius of the smoothing kernel.
        Default value is 30 pixels.
    edge_cut (optional): integer
        Number of pixels to remove from the edges of the image before
        high pass filtering. Default value is 3 pixels.
    growth (optional): float
        Badmap growth factor, used to convolve and grow the badmap.
    buffer_type (optional): string
        Specifies which type of buffer to use, either "files" or
        "frames". Set to "frames" by default. Note that using median
        integration and file buffers are incompatible.
    tolerance (optional): float
        Fraction of available memory to be used for integration.
    threadcount (optional): integer
        Number of threads to employ in multithreading.
        Default value is 50 threads.
    
    Returns: 
    ---------------------- 
    flat: 2D image array
        Stacked flat frame.
    badmap: 2D image array
        Bad pixel map, where bad pixels are set to 0 and all other
        pixels are set to 1.
    filtered_frame: 2D image array
        High pass filtered flat from which the badmap was created.
    """
    
    if buffer_type == "frames":
        flat = hf.integrate_frames_buffer(files, method="mean", tolerance=tolerance,
                                          threadcount=threadcount)
    else:        
        flat = hf.integrate_files_buffer(files, tolerance=tolerance, threadcount=threadcount)

    # Remove zero values from the flat, replace with the minimum value
    flat[flat == 0] = np.min(flat[flat != 0]) 
    smooth_buf = int(1.25*smooth)

    # Create copy image and pad it for convolution
    new_image = np.copy(flat)
    new_image = np.pad(new_image[edge_cut:-1*edge_cut, edge_cut:-1*edge_cut],
                       smooth_buf+edge_cut, mode='edge')

    # High pass filter
    filtered_frame = flat - convolve_fft(new_image, Ring2DKernel(smooth, int(0.8*smooth)))\
                                [smooth_buf:-1*smooth_buf, smooth_buf:-1*smooth_buf]
    
    #filtered_frame[~bools] = np.nan

    # Create badmap by setting pixels above threshold to 0
    badmap = np.ones(np.shape(filtered_frame))
    badmap[(filtered_frame > (sigma*np.std(filtered_frame)+np.median(filtered_frame))) |
           (filtered_frame < (-1*sigma*np.std(filtered_frame)+np.median(filtered_frame)))] = 0

    # Cut out edges, which are often dead columns/rows
    badmap[:edge_cut,:] = 0
    badmap[-1*edge_cut:,:] = 0
    badmap[:,:edge_cut] = 0
    badmap[:,-1*edge_cut:] = 0

    badmap = convolve_fft(badmap, Gaussian2DKernel(growth))
    
    badmap[badmap < 0.9] = 0

    badmap[badmap > 0] = 1
    
    return flat, badmap, filtered_frame

def setup_data(obj, raw_dir, double_side=False, start_frame=None, end_frame = None,
               skip_target_check=False, recalc_para_angles=False, correct_linearity=True,
               frame_median_limit = 28000, cold_stop_crop=0, ncoadds=2, threadcount=50):
    """
    Sets up data by reading parameters from the fits headers and
    creating high pass frames for chop identification. For double sided
    data, the data is split into two sets.
    
    Parameters:
    ----------------------
    obj: string
        Object name, should match the image header
    raw_dir: string/Path object
        Directory where images are read from
    double_side (optional): boolean
        Enable if imaging is double sided. Single sided is the default.
    start_frame (optional): integer
        Frame to start with. By default all files are used.
    end_frame (optional): integer
        Frame to end with. By default all files are used.
    skip_target_check (optional): boolean
        if True, skips checking the image header
        for the target name 'obj'
    recalc_para_angles: boolean
        If True, recalculates parallactic angle
    correct_linearity: boolean
        If True, corrects for detector nonlinearity
    frame_median_limit (optional): integer
        Limit for the median of the frame. Frames with median above this
        limit are rejected. Default is 28000 counts.
    cold_stop_crop (optional): integer
        Number of pixels to mask along the cold stop in
        the case of double-sided imaging.
    ncoadds (optional): integer
        Number of coadded frames in each image.
    threadcount (optional): integer
        Number of threads to employ in multithreading.
        Default value is 50 threads.
        
    Returns:
    ----------------------
    files: 1D numpy array or list of 2 1D numpy arrays
        List of raw file paths, sorted. If double sided, this variable
        is a list containing DX and SX raw file paths.
    chops: 1D numpy array
        List of chop states corresponding to the file list, entries
        are either "CHOP_A" or "CHOP_B"
    header_info: tuple of lists
        Lists of header properties:
        frame_medians: 1D numpy array
            List of frame medians corresponding to the file list.
        para_angles: 1D numpy array
            List of parallactic angles corresponding to the file list.
    """
    
    root_dir = os.path.dirname(raw_dir)
    
    # directory for all files
    files = sorted(list(pathlib.Path(str(raw_dir)).rglob('*.fits')))
    # Get rid of "._" files in macs
    files = np.asarray([a for a in files if a.name[0]!='.' and str(a.parent)==raw_dir])
    
    print("Detected ", len(files), " fits files")

    # Option to test with fewer frames
    files = files[start_frame:end_frame]
        
    print('File count = ', len(files))

    # If data is double sided, split raws into two directories
    if double_side:
        
        sx_raw_dir=os.path.join(root_dir,'sx_'+os.path.basename(raw_dir))
    
        if not os.path.exists(sx_raw_dir):
            os.makedirs(sx_raw_dir)

        dx_raw_dir=os.path.join(root_dir,'dx_'+os.path.basename(raw_dir))
    
        if not os.path.exists(dx_raw_dir):
            os.makedirs(dx_raw_dir)

        new_raw_dirs  = [sx_raw_dir, dx_raw_dir]

    else:

        new_raw_dirs = []

    if correct_linearity:
        
        corrector = hf.LinearityCorrection(ncoadds=ncoadds)
        
        lincorr_raw_dir = os.path.join(root_dir,'lincorr_'+os.path.basename(raw_dir))

        if not os.path.exists(lincorr_raw_dir):
            os.makedirs(lincorr_raw_dir)
            
        if len(new_raw_dirs) != 2:
            new_raw_dirs = [lincorr_raw_dir]
    else:
        corrector = None

    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        
        (chops, frame_medians, para_angles,
         end_times, temps, airmasses,
         wind_spds, wind_dirs, seeing,
         pwvs, exp_times, ncoadds,
         channel_medians, channel_stds) = (
             
         zip(*tqdm(pool.imap(FileInfo((new_raw_dirs, obj, skip_target_check, recalc_para_angles,
                                       frame_median_limit, cold_stop_crop, corrector)),
                             files), total=len(files), desc="Reading file headers"))
        )
        
    if correct_linearity & (len(new_raw_dirs) != 2):
        files = sorted(list(pathlib.Path(str(new_raw_dirs[0])).rglob('*.fits')))
        files = np.asarray([a for a in files if a.name[0]!='.'\
                                       and str(a.parent)==new_raw_dirs[0]])

    chops = np.asarray(chops)
    
    header_info = np.vstack((np.asarray(frame_medians), np.asarray(para_angles), 
                             np.asarray(end_times), np.asarray(temps), np.asarray(airmasses),
                             np.asarray(wind_spds), np.asarray(wind_dirs), np.asarray(seeing),
                             np.asarray(pwvs), np.asarray(exp_times), np.asarray(ncoadds)))

    header_info = np.concatenate((header_info, np.asarray(channel_medians).T,
                                  np.asarray(channel_stds).T))
    
    if double_side:
        sx_raw_files = sorted(list(pathlib.Path(str(sx_raw_dir)).rglob('*.fits')))
        dx_raw_files = sorted(list(pathlib.Path(str(dx_raw_dir)).rglob('*.fits')))
        sx_raw_files = np.asarray([a for a in sx_raw_files if a.name[0]!='.'\
                                   and str(a.parent)==sx_raw_dir])
        dx_raw_files = np.asarray([a for a in dx_raw_files if a.name[0]!='.'\
                                   and str(a.parent)==dx_raw_dir])
        files = [sx_raw_files, dx_raw_files]
    else:
        files = files[~np.isnan(header_info[0])]

    header_info = header_info[:,~np.isnan(header_info[0])]

    return files, chops, header_info

def highpass(files, highpassmask_dir=None, badmap_dir=None, tempflat_dir = None,
             use_temp_flat=True, smooth=30, binsize=10, threadcount=50):
    """
    Sets up data by reading parameters from the fits headers and
    creating high pass frames for chop identification. For double sided
    data, the data is split into two sets.
    
    Parameters:
    ----------------------
    files: 1D numpy array 
        List of raw file paths, sorted.
    highpassmask_dir (optional): string/Path object
        Path to save high pass filtered images
    badmap_dir (optional): 2D Boolean array
        Path to bad pixel map.
    tempflat_dir (optional): 2D numpy array
        Path to temporary flat.
    use_temp_flat (optional): boolean
        Enables the use of a temporary flat. If no directory to a
        temporary flat is given, a new flat is created.
    smooth (optional): integer
        Outer radius of the star mask
    binsize (optional): integer
        Number of pixels for spatial binning.
    threadcount (optional): integer
        Number of threads to employ in multithreading.
        Default value is 50 threads.
        
    Returns:
    ----------------------
    highpass_dir: path
        Directory in which high pass filtered frames are saved.
    """
    
    # Create badmap if unavailable
    if (badmap_dir is None) or (use_temp_flat and tempflat_dir is None):
        tempflat, filtered_frame, badmap = create_new_flat(files)

    # Open badmap if available
    if badmap_dir is not None:
        badmap = (fits.open(badmap_dir))[0].data

    # Use temp flat if needed and available
    if (use_temp_flat and tempflat_dir is not None):
        tempflat = (fits.open(tempflat_dir))[0].data   
    if not use_temp_flat:
        tempflat = None
        
    # Use highpass mask if provided
    if highpassmask_dir is not None:
        highpassmask = (fits.open(highpassmask_dir))[0].data   
        highpassmask = (highpassmask == 1)
    else:
        highpassmask = None

    # Create boolean map from badmap to remove bad pixels
    bools = np.full(np.shape(badmap), False)
    bools[badmap > 0] = True

    
    root_dir =os.path.dirname(os.path.dirname(files[0]))        
    highpass_dir = os.path.join(root_dir,'highpass')
    
    if not os.path.exists(highpass_dir):
        os.makedirs(highpass_dir)
    
    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        results = (zip(*tqdm(pool.imap(HighPass((highpass_dir, bools, highpassmask,
                                                 tempflat, smooth, binsize)), files),
                      total=len(files), desc="Creating high pass filtered frames"))
        )

    return highpass_dir
    
def stellar_chop_identification(files, highpass_dir, nbg=5, chop_direction = 'UP-DOWN'):
    
    """
    Measures the chop states of each file by examining the highpass
    filtered frames and detecting the location of the stellar PSF.

    Parameters:
    ----------------------
    files: 1D numpy array
        List of raw file paths, sorted.
    highpass_dir: path
        Directory where the high pass filtered frames are saved.
    nbg (optional): integer
        Number of frames to use in rolling background subtraction.
        Must be odd, and greater than or equal to 5. Default is 5.
    chop_direction (optional): string
        The chopping direction employed, either "UP-DOWN", "LEFT-RIGHT".
        Default value is "UP-DOWN", used for single-sided imaging.
        
    Returns:
    ----------------------
    chops: 1D numpy array
        List of chop states corresponding to the file list, entries
        are either "CHOP_A" or "CHOP_B"
    maxima: 2D numpy array
        List of coordinates for the maxima after rolling background
        subtraction, hopefully corresponding to the PSF. 
    """
    
    print("Finding chop positions....")
    
    root_dir = os.path.dirname(highpass_dir)

    # Get array shape of highpass data
    hdul = fits.open(os.path.join(highpass_dir, "highpass_"+files[0].name))
    hp_array_shape = np.shape(hdul[0].data)
    hdul.close()

    # Get array shape of raw data
    hdul = fits.open(files[0])
    array_shape = np.shape(hdul[0].data)
    hdul.close()

    # Create array for storing background frames
    frames = np.ones((nbg, hp_array_shape[0], hp_array_shape[1]))

    # Fill array with first nbg images
    for j in range(nbg):
        hdul = fits.open(os.path.join(highpass_dir, "highpass_"+files[j].name))
        frames[j] = hdul[0].data
        hdul.close()

    # Initialize variables
    chops = np.array(["CHOP_A"]*len(files))
    lastChopPosition = ""
    freezeChop = False

    maxima = np.zeros((len(files), 2))

    # Rolling background subtraction
    for i in tqdm(range(len(files))):

        # Update array with new frame
        if (i > int(np.floor(nbg/2))) and (i < (len(files) - int(np.floor(nbg/2)))):
            hdul = fits.open(os.path.join(highpass_dir,
                                          "highpass_"+files[nbg+i - int(np.floor(nbg/2)) -1].name))
            frames = np.concatenate((frames[1:], [hdul[0].data]))
            hdul.close()

        # Calculate new background by taking the minimum of the frames if chop state hasn't frozen
        if not freezeChop:
            bg =  np.min(frames, axis=0)

        # Open image
        image = fits.open(os.path.join(highpass_dir, "highpass_"+files[i].name))

        # Subtract background
        subtracted = image[0].data - bg

        # Find PSF by finding the maximum
        max_indices = np.where(subtracted == np.nanmax(subtracted))

        maxima[i] = max_indices[0][0], max_indices[1][0]
        
        # Calculate chop position based on the position of the star
        if (chop_direction == "UP-DOWN") or (chop_direction == "DIAGONAL"):
            if max_indices[0][0] < hp_array_shape[0]/2:
                chops[i] = "CHOP_A"
            else:
                chops[i] = "CHOP_B"
        elif (chop_direction == "LEFT-RIGHT"):
            if max_indices[1][0] < hp_array_shape[1]/2:
                chops[i] = "CHOP_A"
            else:
                chops[i] = "CHOP_B"     

        # If chop position hasn't changed, indicate that chopping has stopped
        if chops[i] == lastChopPosition:
            freezeChop = True
        else:
            freezeChop = False
        lastChopPosition = chops[i]
        
        image.close()

    maxima = maxima*array_shape[1]/hp_array_shape[0]

    return chops, maxima

def frame_med_chop_identification(orig_frame_medians, files=None,
                                  threshold=0, size=13):
    
    """
    Measures the chop states of each file by using the frame medians.

    Parameters:
    ----------------------
    frame_medians: 1D numpy array
        List of frame medians corresponding to the file list.
    files (optional): 1D numpy array
        List of raw file paths, sorted. If provided, the frame medians
        are recomputed.
    threshold (optional): float
        Threshold for dividing the maximum filtered measurement between
        chop states.
    size (optional): integer
        Parameter for scipy.ndimage.maximum_filter1d, length along
        which to calculate the 1-D maximum.
        
    Returns:
    ----------------------
    chops: 1D numpy array
        List of chop states corresponding to the fmame medians, entries
        are either "CHOP_A" or "CHOP_B"
    frame_medians: 1D numpy array
        List of frame medians corresponding to the file list.
    chopm: 1D numpy array
        List of normalized frame medians corresponding to the file list.
    """
    
    if files is not None:
        frame_medians = np.zeros(len(files))
        for i in tqdm(range(len(files)), desc="Getting frame medians..."):
            hdul = fits.open(files[i])
            frame_medians[i] = np.nanmedian(hdul[0].data[0])
            hdul.close()
    else:
        frame_medians = np.copy(orig_frame_medians)
    
    chops = np.array(["CHOP_A"]*len(frame_medians))

    chopm = frame_medians/maximum_filter1d(frame_medians, size)
    chopm = chopm/np.nanmean(chopm) + threshold*np.std(chopm)

    chops[chopm > 1] = "CHOP_B"

    return chops, frame_medians, chopm

def fourier_chop_identification(files, sigma=1.5, threshold=0.1,
                                size=3, cutout=[], threadcount=50):
    
    """
    Measures the chop states of each file by using the Fourier transform
    of the image.

    Parameters:
    ----------------------
    files: 1D numpy array
        List of raw file paths, sorted.
    sigma (optional): float
        Standard deviation based threshold to discard the
        low frequency end of the Fourier transform of the image.
    threshold (optional): float
        Threshold for dividing the maximum filtered measurement between
        chop states.
    size (optional): integer
        Parameter for scipy.ndimage.maximum_filter1d, length along
        which to calculate the 1-D maximum.
    threadcount (optional): integer
        Number of threads to employ in multithreading.
        Default value is 50 threads.
    Returns:
    ----------------------
    chops: 1D numpy array
        List of chop states corresponding to the fmame medians, entries
        are either "CHOP_A" or "CHOP_B"
    measures: 1D numpy array
        List of fourier means corresponding to the file list.
    chopm: 1D numpy array
        List of normalized frame medians corresponding to the file list.
    """

    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        measures, indices =\
            zip(*tqdm(pool.imap(FourierMean((files, sigma, cutout)),
                                range(len(files))),
                      desc="Calculating fourier means", total=(len(files))))

    
    chops = np.array(["CHOP_A"]*len(measures))

    chopm = measures/maximum_filter1d(measures, size)
    chopm = chopm/(np.nanmedian(chopm) + threshold*np.std(chopm))

    chops[chopm > 1] = "CHOP_B"

    return chops, measures, chopm
    
def chop_correction(orig_files, orig_chops, orig_header_info,
                    orig_maxima=None, coadd_limit = 10):
    
    """
    Adjacent frames at the same chop state are coadded, and the
    parallactic angles and maxima are matched accordingly.
    
    Parameters:
    ----------------------
    orig_files: 1D numpy array
        List of raw file paths, sorted.
    orig_chops: 1D numpy array
        List of chop states corresponding to the file list, entries
        are either "CHOP_A" or "CHOP_B"
    orig_header_info: tuple of lists
        Lists of header properties:
        frame_medians: 1D numpy array
            List of frame medians corresponding to the file list.
        para_angles: 1D numpy array
            List of parallactic angles corresponding to the file list.
    orig_maxima (optional): 2D numpy array
        List of coordinates for the maxima after rolling background
        subtraction, hopefully corresponding to the PSF. 
    coadd_limit (optional): integer
        Maximum number of frames that can be coadded. Default is 10.
 
    Returns:
    ----------------------
    files: 1D numpy array
        List of raw file paths, sorted and accounting for
        coadded frames.
    chops: 1D numpy array
        List of measured chop states corresponding to the file list,
        sorted and accounting for coadded frames.
    para_angles: 1D numpy array
        List of parallactic angles corresponding to the file list,
        sorted and accounting for coadded frames.
    maxima: 2D numpy array
        List of coordinates for the maxima after rolling background
        subtraction, hopefully corresponding to the PSF.
    header_info: tuple of lists
        Lists of header properties corresponding to the file list
    """

    root_dir = os.path.dirname(os.path.dirname(orig_files[0]))

    # Create directory for coadded frames
    coadd_dir=os.path.join(root_dir,'coadd')
    
    if not os.path.exists(coadd_dir):
        os.makedirs(coadd_dir)
        
    # Create copies of these arrays to avoid overwriting them
    files = np.copy(orig_files)
    chops = np.copy(orig_chops)
    header_info = np.copy(orig_header_info)
    
    if orig_maxima is None:
        maxima = None
    else:
        maxima = np.copy(orig_maxima)

    print("Finding consecutive repeat chop positions...")

    # Group chops together if they have the same value and are consecutive using groupby
    chop_groups = []
    k = 0

    for i, j in tqdm(groupby(chops)):
        sum = (len(list(j)))

        # Only include groups with more than one chop
        if sum != 1:
            chop_groups.append((k, sum))
        k += sum
    
    print("Coadding consecutive repeat chop positions...")

    # Get array shape of data
    hdul = fits.open(files[0])
    array_shape = np.shape(hdul[0].data)
    hdul.close()

    bools = np.full(len(files), True)
    # Go through groups
    for group in tqdm(chop_groups):
        
        frames = np.zeros(array_shape)
        count = 0

        # Read every image in group until coadd limit is reached and add them
        for i in range(group[1]):
            hdul = fits.open(files[group[0]+i])
            if (i < coadd_limit):
                count += 1
                frames += hdul[0].data
            hdul.close()

            # Set all entries past those of the first image to flagged values for deletion
            if i != 0:
                
                #files[group[0]+i] = ''
                #chops[group[0]+i] = ''
                bools[group[0]+i] = False

                header_info[:,group[0]] = ((header_info[:,group[0]]*i +
                                            header_info[:,group[0]+i]) / (i+1))
                header_info[-1,group[0]] *= i+1
                #header_info[group[0]+i] = np.full(np.shape(header_info[group[0]+i]), np.nan)
                
                if maxima is not None:
                    maxima[group[0]+i] = 0, 0
                    
        newhdul = fits.HDUList([fits.PrimaryHDU(data=frames/count)])
        files[group[0]] = pathlib.Path(coadd_dir, files[group[0]].name)
        newhdul.writeto(files[group[0]], overwrite=True)
        newhdul.close()

    if maxima is not None:
        maxima = maxima[bools]

    return files[bools], chops[bools], header_info[:,bools], maxima

def chop_finder(orig_files, orig_para_angles, directory, metrics=["locdiff"],
                chop_direction = 'UP-DOWN', coadd_limit = 10, threadcount=50):

    """
    Measures the chop states of each file by examining input frames and
    employing various metrics to determine the chop state.
    Adjacent frames at the same chop state are coadded, and the
    parallactic angles are matched accordingly.
    
    Parameters:
    ----------------------
    orig_files: 1D numpy array
        List of raw file paths, sorted.
    directory: path
        Directory where the input images are located.
    metrics (optional): list
        List of metrics to use to determine chop state:
            "minmax":
                Difference between the minimum and maximum of the frame
                after chop subtraction.
            "stdev":
                Standard deviation of the frame after chop subtraction.
            "bgmean":
                Mean of the absolute value of the frame after
                chop subtraction.
            "locdiff":
                Difference in position vector between
                the maximum and minimum of the frame.
                
    chop_direction (optional): string
        The chopping direction employed, either "UP-DOWN", "LEFT-RIGHT".
        Default value is "UP-DOWN", used for single-sided imaging.
    coadd_limit (optional): integer
        Maximum number of frames that can be coadded. Default is 10.
    threadcount (optional): integer
        Number of threads to employ in multithreading.
        Default value is 50 threads.
        
    Returns:
    ----------------------
    files: 1D numpy array
        List of raw file paths, sorted and accounting for
        coadded frames.
    chops: 1D numpy array
        List of chop states corresponding to the file list,
        sorted and accounting for coadded frames, measured by detecting
        the change in chop states using the metrics.
    para_angles: 1D numpy array
        List of parallactic angles corresponding to the file list,
        sorted and accounting for coadded frames.
    chopstate_derived: 1D numpy array
        List of chop states corresponding to the file list,
        sorted and accounting for coadded frames, measured by using
        the calculated position difference.
    """
    
    # Create copies of these arrays to avoid overwriting them
    files = np.copy(orig_files)
    para_angles = np.copy(orig_para_angles)
    
    print("Finding chop positions....")
    
    root_dir = os.path.dirname(directory)

    # Create directory for saving coadded images
    coadd_dir=os.path.join(root_dir,'coadd')   
    if not os.path.exists(coadd_dir):
        os.makedirs(coadd_dir)

    # Get shape of frames
    hdul = fits.open(os.path.join(directory, files[0].name))
    array_shape = np.shape(hdul[0].data)

    # Handle raws
    if hdul[0].data.ndim == 3:
        array_shape = (array_shape[1], array_shape[2])
    hdul.close()

    # Initialize arrays for metrics
    minmax, locdiff, stdev, bgmean = (np.zeros(len(files) - 1), np.zeros(len(files) - 1),
                                      np.zeros(len(files) - 1), np.zeros(len(files) - 1))

    # Initialize arrays for chop states
    chop_guesses, chopstate_derived = (np.zeros((len(files)), dtype='<U6'), 
                                       np.zeros((len(files)), dtype='<U6'))
    maxima = np.zeros(((len(files)-1), 2))
    
    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        minmax, stdev, bgmean, locdiff, maxima, chop_guesses, indices =\
            zip(*tqdm(pool.imap(ChopMetrics((files, directory, array_shape, chop_direction)),
                                range(len(files)-1)),
                      desc="Calculating chop metrics", total=(len(files)-1)))
        
    # Normalize all metrics to the mean, convert to numpy arrays
    minmax = np.asarray(minmax/np.mean(minmax))
    locdiff = np.asarray(locdiff/np.mean(np.abs(locdiff)))
    stdev = np.asarray(stdev/np.mean(stdev))
    bgmean = np.asarray(bgmean/np.mean(bgmean))
    chop_guesses = np.asarray(chop_guesses)
    print(chop_guesses)

    # Normalization with reordering (somehow not needed)
    '''
    minmax = np.take(minmax/np.mean(minmax), indices)
    locdiff = np.take(locdiff/np.mean(np.abs(locdiff)), indices)
    stdev = np.take(stdev/np.mean(stdev), indices)
    bgmean = np.take(bgmean/np.mean(bgmean), indices)
    chop_guesses = np.take(chop_guesses, indices)
    '''

    # Locate first change in chop state
    first_change = np.where(np.abs(locdiff) > 1)[0][0]

    # Identify chops before the first change based on position difference
    if (locdiff[np.where(np.abs(locdiff) > 1)[0][0]] < 0):
        
        chopstate_derived[:first_change+1] = "CHOP_A"

    else:
        
        chopstate_derived[:first_change+1] = "CHOP_B"    

    # Array to combine metrics
    finalcomp = np.zeros(len(files)-1)
    count = 0
    
    if "minmax" in metrics:
        finalcomp += minmax
        count += 1
    if "locdiff" in metrics:
        finalcomp += np.abs(locdiff)
        count += 1
    if "stdev" in metrics:
        finalcomp += stdev
        count += 1
    if "bgmean" in metrics:
        finalcomp += bgmean
        count += 1

    # Combined metric array
    finalcomp = finalcomp/count

    # Determine chops by using recorded changes in chop state
    for i in range(len(files)-1):
        if finalcomp[i] < 1:
            chopstate_derived[i+1] = chopstate_derived[i]
        else:
            if chopstate_derived[i] == "CHOP_A":
                
                chopstate_derived[i+1] = "CHOP_B"
                
            else:
                
                chopstate_derived[i+1] = "CHOP_A"

    # Check to make sure both methods of chop determination agree
    if len(chop_guesses[chop_guesses != chopstate_derived[:-1]]) > 0:
        print(chop_guesses != chopstate_derived[:-1])        
        raise ValueError("Chop mismatch")

    return files, chops, para_angles, chopstate_derived
