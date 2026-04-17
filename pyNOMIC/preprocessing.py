#----------------------------------------
# IMPORTS
#----------------------------------------

import numpy as np
from itertools import groupby
import os, pathlib, psutil
from astropy.io import fits
from tqdm.auto import tqdm
from astropy.convolution import convolve_fft,\
    Box2DKernel, Ring2DKernel
from scipy.signal import correlate
from scipy.optimize import curve_fit
from image_registration import chi2_shift
from multiprocessing.pool import ThreadPool as Pool
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
            Path to save binned image, if value is None the image is returned instead of saved
        """
        
        self.params = params
        self.integrator = hf.IntegrateFrames(params[0])
    
    def __call__(self, angles_files_tuple):

        """
        Parameters:
        ----------------------
        angles_files_tuple: tuple containing 'angles' and 'files'

            angles: list or array
                List of parallactic angles corresponding to each file in 'files'
                
            files: list or array 
                List of raw file paths, sorted 
             
        Returns: 
        ---------------------- 
        binned frame: 2D image array
            binned frame (averaged), only returned if binned_dir is None

        mean_parallactic_angle: float
            Average parallactic angle over the frames

        filename: string
            The filename of the first file in the bin, with the prefix "binned_" added on
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

class FileInfoHighPass(object):

    '''
    Open raw images and obtain information from fits header and create high pass filtered images.
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
            mask applied to images before high pass filtering, affected pixels are set to np.nan
        tempflat: 2D numpy array
            Temporary flat applied to locate the star
        obj: string
            Object name, should match the image header
        skip_target_check: boolean
            if True, skips checking the image header for the target name 'obj'
        smooth: integer
            Outer radius of the star mask
        new_raw_dirs: list
            List containing two directory paths for spliting the raw data in the case of double-sided imaging.
            Is set to None in the case of single-sided imaging.
        cold_stop_crop: integer
            Number of pixels to mask along the cold stop in the case of double-sided imaging.
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

        Returns:
        ----------------------    
        chop: string
            The chop state identified from the image header
        frame_median: integer
            The median value of the frame
        para_angle: float
            The parallactic angle identified from the image header
        """
        
        highpass_dir, bools, highpassmask, tempflat, obj, skip_target_check, smooth, new_raw_dirs, cold_stop_crop, binsize = self.params

        hdul = fits.open(file)

        orig = hdul[0].data[0]

        # If flat is available, create flat divided image
        if tempflat is not None:
            image = orig/tempflat
        else:
            image = np.copy(orig)

        # Check if correct object
        if not skip_target_check:
            try:
                object = hdul[0].header['OBJNAME']
            except:
                print(file)
                raise ValueError("OBJNAME header not found")
            if object != obj:
                raise ValueError("Object in header does not match given object")

        # Obtain other fits header information (currently commented out)
        '''        
        time = Time(hdul[0].header['DATE-OBS'] +"T"+ hdul[0].header['TIME-END'], format='isot', scale='utc')
        
        end_time = time.jd
        temp = hdul[0].header['LBTTEMP']
        airmass = hdul[0].header['LBT_AIRM']
        wind_spd = hdul[0].header['WINDSPD']
        wind_dir = hdul[0].header['WINDDIR']
        seeing = hdul[0].header['SEEING']
        pwv = hdul[0].header['SMTTAU']
        exp_time = hdul[0].header['EXPTIME']
        ncoadds = hdul[0].header['NCOADDS']
        #nod_pos, dettemp, nomiccfw not found
        '''

        # Obtain chop state if available
        try:
            chop = hdul[0].header['CHOP_POS']
        except:
            chop = "CHOP_NA"

        # Obtain parallactic angle
        para_angle = float(hdul[0].header['LBT_PARA'])

        # Get frame median
        frame_median = np.median(hdul[0].data[0])

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
        filtered_frame = convolve_fft(image - convolve_fft(new_image, Box2DKernel(smooth))[smooth_buf:-1*smooth_buf, smooth_buf:-1*smooth_buf], Box2DKernel(binsize*3))
        
        # Do if data is double sided
        if len(new_raw_dirs) == 2:
            
            array_shape = np.shape(filtered_frame)

            sx_filtered_frame = filtered_frame[:int(0.5*array_shape[0]-cold_stop_crop), :]
            dx_filtered_frame = filtered_frame[cold_stop_crop+int(0.5*array_shape[0]):, :]
            
            sx_filtered_frame,_ = hf.spatial_binning([sx_filtered_frame], binsize)
            dx_filtered_frame,_ = hf.spatial_binning([dx_filtered_frame], binsize)   
            
            # Write high pass images to path
            newhdul = fits.HDUList([fits.PrimaryHDU(data=sx_filtered_frame[0])])
            newhdul.writeto(os.path.join(highpass_dir[0], "highpass_"+file.name[:-5]+"_sx.fits"), overwrite=True)
            newhdul.close()
            
            newhdul = fits.HDUList([fits.PrimaryHDU(data=dx_filtered_frame[0])])
            newhdul.writeto(os.path.join(highpass_dir[1], "highpass_"+file.name[:-5]+"_dx.fits"), overwrite=True)
            newhdul.close()

            # Write split raw images to path
            newhdul = fits.HDUList([fits.PrimaryHDU(data=[orig[:int(0.5*array_shape[0]-cold_stop_crop), :]])])
            newhdul.writeto(os.path.join(new_raw_dirs[0], file.name[:-5]+"_sx.fits"), overwrite=True)
            newhdul.close()

            newhdul = fits.HDUList([fits.PrimaryHDU(data=[orig[cold_stop_crop+int(0.5*array_shape[0]):, :]])])
            newhdul.writeto(os.path.join(new_raw_dirs[1], file.name[:-5]+"_dx.fits"), overwrite=True)
            newhdul.close()
            
        else:

            filtered_frame,_ = hf.spatial_binning([filtered_frame], binsize)   
            
            # Write high pass image to path
            newhdul = fits.HDUList([fits.PrimaryHDU(data=filtered_frame[0])])
            newhdul.writeto(os.path.join(highpass_dir, "highpass_"+file.name), overwrite=True)
            newhdul.close()
            
        hdul.close()

        return chop, frame_median, para_angle

class ChopMetrics(object):

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
        
    def __call__(self, i):

        """
        Parameters:
        ----------------------
        index: integer
            File index to process from 'files'

        Returns: 
        ---------------------- 
        stacked_img: 2D image array
            Integrated stack of frames (sum)
            
        count: integer
            Number of frames
        """     

        files, directory, array_shape, chop_direction = self.params
        
        image = fits.open(os.path.join(directory, files[i].name))

        image2 = fits.open(os.path.join(directory, files[i+1].name))

        subtract =  convolve(image[0].data[0] - image2[0].data[0],  Box2DKernel(10))
                
        image.close()
        image2.close()

        minmax = np.max(subtract) - np.min(subtract)
        stdev = np.std(subtract)
        bgmean = np.mean(np.abs(subtract))
        
        indices = np.where(subtract == np.nanmax(subtract))
        
        if (chop_direction == "UP-DOWN") or (chop_direction == "DIAGONAL"):
            
            locdiff = np.where(subtract == np.max(subtract))[0][0] - np.where(subtract == np.min(subtract))[0][0]

            if indices[0][0] < array_shape[0]/2:
                chop_guess = "CHOP_A"
            else:
                chop_guess = "CHOP_B"
                
        elif (chop_direction == "LEFT-RIGHT"):
            
            locdiff = np.where(subtract == np.max(subtract))[1][0] - np.where(subtract == np.min(subtract))[1][0]

            if indices[1][0] < array_shape[1]/2:
                chop_guess = "CHOP_A"
            else:
                chop_guess = "CHOP_B"     
            
        maximum = [indices[0][0], indices[1][0]]

        return minmax, stdev, bgmean, locdiff, maximum, chop_guess, i

class SubtractBackground(object):

    '''
    Subtract background from adjacent chop frames, divide by flat, perform highpass filter
    '''

    def __init__(self, params):

        """
        Parameters (contained inside a tuple):
        ----------------------
        subtracted_dir: string/Path object
            Directory where images will be saved
        raw_files: list or array 
            List of raw file paths, sorted 
        psf_subtracted_files: list or array 
            List of psf subtracted file paths, sorted 
        chops: string array
            List of chop states corresponding to the file list, entries are either "CHOP_A" or "CHOP_B"
        edge_cut: integer
            Number of pixels to remove at the edges of images
        channel_edges: list
            List containing indices corresponding to channel edges
        biased_columns: list
            List containing indices corresponding to biased columns
        striped_regions: list
            List containing indices corresponding to regions of the image needing destriping. Each entry contains four integers for slicing the image: [0:1, 2:3]
        vertical_biases: list
            List containing indices corresponding to columns separating regions of the image with different biases
        horizontal_biases: list
            List containing indices corresponding to rows separating regions of the image with different biases
        biased_rows: list
            List containing indices corresponding to biased rows
        nanrows: list
            List containing indices corresponding to rows that need to be set to np.nan
        nancols: list
            List containing indices corresponding to columns that need to be set to np.nan
        flats: list
            List containing the flats for each chop state
        nbg: integer
            Number of frames to use in rolling background subtraction
        smooth: integer
            Radius of smoothing kernel, divided by 5
        """
        
        self.params = params
        
    def __call__(self, i):

        """
        Parameters:
        ----------------------
        index: integer
            File index to process from 'files'
        """     

        subtracted_dir, raw_files, psf_subtracted_files, chops, edge_cut, channel_edges, biased_columns, striped_regions,\
            vertical_biases, horizontal_biases, biased_rows, nanrows, nancols, flats, nbg, smooth = self.params

        # Locate adjacent images of the same chop state
        bg_indices = np.arange(i - 2*nbg + 1, i + 2*nbg, 2)
        # Exclude nonexistent indices and the current index
        bg_indices = bg_indices[(bg_indices >= 0) & (bg_indices < len(raw_files)) & (bg_indices != i)]

        # Open image and get array shape
        unsubtracted = fits.open(raw_files[i])
        array_shape = np.shape(unsubtracted[0].data[0])

        # Create background image from adjacent images (different chop state)    
        bg = np.zeros(array_shape)
        for j in bg_indices:
            hdul = fits.open(psf_subtracted_files[j])
            bg += hdul[0].data
            hdul.close()
            
        # Averaged background
        bg = bg / len(bg_indices)

        # Use appropriate flat depending on the chop state, adjacent image is of a different chop
        if chops[i] == "CHOP_A":
            # Flat correction
            subtracted_frame = (unsubtracted[0].data[0]/flats[0] - bg/flats[1])

        else:
            # Flat correction
            subtracted_frame = (unsubtracted[0].data[0]/flats[1] - bg/flats[0])   

        # Zero point the frame
        subtracted_frame -= np.nanmedian(subtracted_frame)

        # These steps are done in a very particular order
        # Remove channel edges
        for channel_edge in channel_edges:
            subtracted_frame = hf.repairChannelEdges(subtracted_frame, channel_edge)

        # Remove vertical lines
        for biased_column in biased_columns:
            subtracted_frame = hf.repairVerticalLine(subtracted_frame, biased_column)

        # Destripe
        for striped_region in striped_regions:
            subtracted_frame[striped_region[0]:striped_region[1], striped_region[2]:striped_region[3]] =\
                hf.destriping(subtracted_frame[striped_region[0]:striped_region[1], striped_region[2]:striped_region[3]],\
                           ref=striped_region[4])

        # Remove vertical biases
        for vertical_bias in vertical_biases:
            subtracted_frame = hf.repairVerticalBias(subtracted_frame, vertical_bias)

        # Remove horizontal biases
        for horizontal_bias in horizontal_biases:
            subtracted_frame = hf.repairHorizontalBias(subtracted_frame, horizontal_bias)

        # Remove horizontal lines
        for biased_row in biased_rows:
            subtracted_frame = hf.repairHorizontalLine(subtracted_frame, biased_row)

        # Set rows and columns to nan if needed
        for loc in nanrows:
            subtracted_frame[loc, :] = np.nan
        for loc in nancols:
            subtracted_frame[:, loc] = np.nan

        # Remove image edges as they are often bad columns/rows
        subtracted_frame = subtracted_frame[edge_cut:-1*edge_cut ,edge_cut :-1*edge_cut ]

        # Adjust array shape with edge removal
        array_shape = (array_shape[0] - 2*edge_cut, array_shape[1] - 2*edge_cut)

        # Locate PSF by finding the maximum
        max_indices = np.where(subtracted_frame == np.nanmax(subtracted_frame))

        maximum = (max_indices[1][0], max_indices[0][0])

        # Create removal mask for both the PSF and oversubtracted PSF
        mask = hf.psf_removal_mask((maximum[0], maximum[1]), 32, 35, array_shape[1], array_shape[0])

        # Create background model, use mask to mask out PSF
        new_bg = np.copy(subtracted_frame)
        new_bg = new_bg*(1-mask) + np.nanmedian(new_bg)*(mask)

        # Subtract convolved background
        subtracted_frame = subtracted_frame - convolve_fft(np.pad(new_bg, 50, mode='edge'), Ring2DKernel(5*smooth, 4*smooth))[50:-50, 50:-50]

        # Write image to file
        newhdul = fits.HDUList([fits.PrimaryHDU(data=(subtracted_frame))])
        newhdul.writeto(os.path.join(subtracted_dir, "subtracted_"+raw_files[i].name), overwrite=True)
        newhdul.close()
    
        unsubtracted.close()

        # Return minimum and maximum locations
        return maximum, True

class RegisterFrames(object):

    """
    Aligns frames using both a PSF reference and a provided PSF location, then pads the image as necessary.
    """

    def __init__(self, params):

        """
        Parameters (contained inside a tuple):
        ----------------------
        subtracted_dir: string/Path object
            Directory where images are read from
        aligned_dir: string/Path object
            Directory where aligned images will be saved
        padding: integer tuple
            Tuple describing number of nan columns and rows to add to the image when aligning
        center_padding: integer tuple
            Tuple describing number of nan columns and rows to add to the image to place the PSF at the center of the image
        px: 1D numpy array
            Array enumerating columns of the alignment grid
        py: 1D numpy array
            Array enumerating rows of the alignment grid
        reference: 2D numpy array
            Reference image cutout for subpixel PSF alignment
        first_maxima: float tuple
            Tuple encoding location of the PSF in the original reference image (the first image of the cube)
        windowsize: integer
            Half width/height of the reference cutout image (which is 1:1 aspect ratio)
        nan_mask_diameter: integer
            Diameter of the mask used to mask out the oversubtracted/negative image PSF
        Returns:
        ----------------------    
        offsets: tuple
            Tuple containing the total offset of the PSF from the original reference image.
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
        offsets: tuple
            Tuple containing the total offset of the PSF from the original reference image.
        """

        subtracted_dir, aligned_dir, padding, center_padding,\
        px, py, reference, first_maxima, windowsize, nan_mask_diameter = self.params

        # Open file
        hdul = fits.open(os.path.join(subtracted_dir, "subtracted_"+file.name))
        frame = hdul[0].data
        hdul.close()
        
        # Locate PSF by finding the maximum and the oversubtracted PSF by finding the minimum
        maxima = np.where(frame == np.nanmax(frame))
        minima = np.where(frame == np.nanmin(frame))

        # Create cutout of PSF to save resources
        cutout = frame[(maxima[0][0]-windowsize):(maxima[0][0]+windowsize),(maxima[1][0]-windowsize):(maxima[1][0]+windowsize)]

        # Find offset of PSF relative to maxima/minima
        try:
            offset = chi2_shift(reference, cutout, upsample_factor='auto', return_error=False)
        except:
            offset = [0,0]

        # Add offset of maxima from the first image to compute total offset
        offset[0] += (maxima[1][0] - first_maxima[1][0])
        offset[1] += (maxima[0][0] - first_maxima[0][0])

        '''
        # Mask out the oversubtracted PSF
        frame[hf.circular_mask((minima[1][0], minima[0][0]), nan_mask_diameter, len(px)-np.abs(padding[0]), len(py)-np.abs(padding[1]))] = np.nan
        '''
        frame = hf.align_frame(frame, px, py, padding, offset)
        frame = hf.pad_frame(frame, len(px) + np.abs(center_padding[0]), len(py) + np.abs(center_padding[1]), center_padding)

        # Write image to file
        newhdul = fits.HDUList([fits.PrimaryHDU(data=(frame))])
        newhdul.writeto(os.path.join(aligned_dir, "aligned_"+file.name), overwrite=True)
        newhdul.close()

        return (offset[0], offset[1]), file

class EvaluateFrames(object):
    
    def __init__(self, params):

        """
        Parameters (contained inside a tuple):
        ----------------------
        img_files: string/Path object
            Directory where images are read from
        chopa_mean_frame: 2D image array
            Stacked image frame for CHOP_A
        chopb_mean_frame: 2D image array
            Stacked image frame for CHOP_B
        wx: 1D numpy array
            Array enumerating columns of the alignment grid
        wy: 1D numpy array
            Array enumerating rows of the alignment grid
        windowsize: integer
            Half width/height of the reference cutout image (which is 1:1 aspect ratio)
        array_shape: integer tuple
            Tuple containing image dimensions, from numpy.shape
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
            Maximum value of the cross correlation of the frame and its respective mean frame.
        amplitude: float
            Amplitude of the Gaussian fit of the PSF.
        sigmax: float
            Standard deviation along x of the Gaussian fit of the PSF.
        sigmay: float
            Standard deviation along y of the Gaussian fit of the PSF.
        gauss_offset: float
            Offset of the gaussian fit of the PSF.
        """
        
        img_files, chopa_mean_frame, chopb_mean_frame, wx, wy, windowsize, array_shape = self.params

        chop, file = chop_file_tuple

        # if an array of files is not provided, file is a path
        if img_files is None:
            hdul = fits.open(file)
            frame = hdul[0].data
            hdul.close()
        # if an array of files is provided, file is an index
        else:
            frame = img_files[int(file)]
            
        # Get the value of the maximum
        psfmaxima = np.nanmax(frame)

        # Compute background deviation by excluding values 3 sigma above the image median
        background_dev = np.nanstd(frame[np.abs(frame - np.nanmedian(frame)) < 3*np.nanstd(frame)])

        # Remove all nans for cross correlation, replace with 0s
        frame[np.isnan(frame)] = 0

        # Use corresponding average frame based on chop state
        if chop == "CHOP_B":
            corr = (np.max(correlate(chopb_mean_frame, frame)))
        else:
            corr = (np.max(correlate(chopa_mean_frame, frame)))    

        # Create image cutout for airy fitting
        cutout = frame[(int(array_shape[0]/2)-windowsize):(int(array_shape[0]/2)+windowsize),(int(array_shape[1]/2)-windowsize):(int(array_shape[1]/2)+windowsize)]
        
        try:
            # Run curve_fit to get airy best fit parameters
            reffit, _ = curve_fit(hf.airydisk_ravel, (wx, wy), cutout.ravel(), p0=[np.max(cutout),\
                        10,10, -1, 0, windowsize, windowsize],\
                       bounds=([0, 1, 1, 1*-np.inf, 0, 1, 1], [2*np.max(cutout), 200, 200, np.inf, 2*np.pi, 2*windowsize, 2*windowsize]))
        except:
            
            return psfmaxima, background_dev, corr, np.nan, np.nan, np.nan, np.nan
            
        return psfmaxima, background_dev, corr, reffit[0], reffit[1], reffit[2], reffit[3]

#----------------------------------------
# FUNCTIONS
#----------------------------------------

def setup_data(obj, raw_dir, highpassmask_dir=None, badmap_dir=None, tempflat_dir = None, testing=False, test_number=None, start_frame=None,\
            end_frame = None, skip_target_check=False, background_limit = 28000, threadcount=50, smooth=30, use_temp_flat=True, double_side=False, cold_stop_crop=0, binsize=10):

    """
    Parameters:
    ----------------------
    Returns:
    ----------------------    
    """
    
    root_dir = os.path.dirname(raw_dir)
    
    # directory for all files
    files = sorted(list(pathlib.Path(str(raw_dir)).rglob('*.fits')))
    
    print("Detected ", len(files), " fits files")
        
    print('Start frame =',start_frame)

    # Option to test with fewer frames
    if testing:
        if test_number is None:
            raise ValueError("If testing, you must specify the number of frames to test in the test_number keyword.")
        files=files[start_frame:start_frame+int(test_number)]
    else:
        files = files[start_frame:end_frame]
        
    print('New file count = ', len(files))
    
    print('Reading file headers and creating high pass filtered frames...')

    # Prepare chops and median arrays
    chops, frame_medians = [], []

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

    # If data is double sided, split raws into two directories
    if double_side:
        
        sx_highpass_dir=os.path.join(root_dir,'sx_highpass')
    
        if not os.path.exists(sx_highpass_dir):
            os.makedirs(sx_highpass_dir)

        dx_highpass_dir=os.path.join(root_dir,'dx_highpass')
    
        if not os.path.exists(dx_highpass_dir):
            os.makedirs(dx_highpass_dir)

        sx_raw_dir=os.path.join(root_dir,'sx_raw')
    
        if not os.path.exists(sx_raw_dir):
            os.makedirs(sx_raw_dir)

        dx_raw_dir=os.path.join(root_dir,'dx_raw')
    
        if not os.path.exists(dx_raw_dir):
            os.makedirs(dx_raw_dir)

        highpass_dir = [sx_highpass_dir, dx_highpass_dir]
        new_raw_dirs  = [sx_raw_dir, dx_raw_dir]

    else:
        
        highpass_dir=os.path.join(root_dir,'highpass')
        new_raw_dirs = []
    
        if not os.path.exists(highpass_dir):
            os.makedirs(highpass_dir)
    
    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        chops, frame_medians, para_angles = zip(*tqdm(pool.imap(FileInfoHighPass((highpass_dir, bools, highpassmask, tempflat, obj,\
                                                skip_target_check, smooth, new_raw_dirs, cold_stop_crop, binsize)), files), total=len(files)))
    
    if double_side:
        sx_raw_files = np.asarray(sorted(list(pathlib.Path(str(sx_raw_dir)).rglob('*.fits'))))
        dx_raw_files = np.asarray(sorted(list(pathlib.Path(str(dx_raw_dir)).rglob('*.fits'))))
        files = [sx_raw_files, dx_raw_files]
    else:
        files = np.asarray(files)

    return files, np.asarray(chops), np.asarray(frame_medians), np.asarray(para_angles), highpass_dir

def chop_correction(orig_files, highpass_dir, orig_chops, orig_para_angles, nbg=5, coadd_limit = 10,\
                    chop_direction = 'UP-DOWN', set_chop_via_bg=False):
    #NBG see above must be odd, greater than or equal to 5

    # Create copies of these arrays to avoid overwriting them
    files = np.copy(orig_files)
    chops = np.copy(orig_chops)
    para_angles = np.copy(orig_para_angles)
    
    print("Finding chop positions....")
    
    root_dir = os.path.dirname(highpass_dir)

    # Create directory for coadded frames
    coadd_dir=os.path.join(root_dir,'coadd')
    
    if not os.path.exists(coadd_dir):
        os.makedirs(coadd_dir)

    # Get array shape of data
    hdul = fits.open(os.path.join(highpass_dir, "highpass_"+files[0].name))
    hp_array_shape = np.shape(hdul[0].data)
    hdul.close()

    # Create array for storing background frames
    frames = np.ones((nbg, hp_array_shape[0], hp_array_shape[1]))

    # Fill array with first nbg images
    for j in range(nbg):
        hdul = fits.open(os.path.join(highpass_dir, "highpass_"+files[j].name))
        frames[j] = hdul[0].data
        hdul.close()

    # Initialize variables
    lastChopPosition = ""
    chopFreeze = False

    maxima, minima = np.zeros((len(files), 2)), np.zeros((len(files), 2))

    # Rolling background subtraction
    for i in tqdm(range(len(files))):

        # Update array with new frame
        if (i > int(np.floor(nbg/2))) and (i < (len(files) - int(np.floor(nbg/2)))):
            hdul = fits.open(os.path.join(highpass_dir, "highpass_"+files[nbg+i - int(np.floor(nbg/2)) -1].name))
            frames = np.concatenate((frames[1:], [hdul[0].data]))
            hdul.close()

        # Calculate new background by taking the minimum of the frames if chop state hasn't frozen
        if chopFreeze == False:
            bg =  np.min(frames, axis=0)

        # Open image
        image = fits.open(os.path.join(highpass_dir, "highpass_"+files[i].name))

        # Find minimum for future use
        min_indices = np.where(image[0].data == np.nanmin(image[0].data))
        minima[i] = min_indices[0][0], min_indices[1][0]

        # Subtract background
        subtracted = image[0].data - bg

        # Find PSF by finding the maximum
        max_indices = np.where(subtracted == np.nanmax(subtracted))

        maxima[i] = max_indices[0][0], max_indices[1][0]
        
        if not set_chop_via_bg:

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
        else:
            raise ValueError("Code for setting chop via background is not written yet")

        # If chop position hasn't changed, indicate that chopping has stopped
        if chops[i] == lastChopPosition:
            chopFreeze = True
        else:
            chopFreeze = False
        lastChopPosition = chops[i]
        
        image.close()

    orig_chops = np.copy(chops)

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

    # Go through groups
    for group in tqdm(chop_groups):
        
        frames = np.zeros((1, array_shape[1], array_shape[2]))
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
                files[group[0]+i] = ''
                chops[group[0]+i] = ''
                para_angles[group[0]+i] = 500
                maxima[group[0]+i] = 0, 0
                minima[group[0]+i] = 0, 0
        newhdul = fits.HDUList([fits.PrimaryHDU(data=frames/count)])
        files[group[0]] = pathlib.Path(coadd_dir, files[group[0]].name)
        newhdul.writeto(files[group[0]], overwrite=True)
        newhdul.close()
        
    files = np.delete(files, np.where(np.asarray(files) == '')[0])
    chops = np.delete(chops, np.where(chops == '')[0])
    para_angles = np.delete(para_angles, np.where(para_angles == 500)[0])
    maxima = np.delete(maxima, np.where(maxima == 0)[0], 0)*array_shape[1]/hp_array_shape[0]
    minima = np.delete(minima, np.where(minima == 0)[0], 0)*array_shape[1]/hp_array_shape[0]

    return files, chops, para_angles, maxima, minima, orig_chops

def chop_finder(orig_files, directory, orig_para_angles, metrics=["locdiff"], coadd_limit = 10,\
                    chop_direction = 'UP-DOWN', threadcount=50):

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
    minmax, locdiff, stdev, bgmean = np.zeros(len(files) - 1), np.zeros(len(files) - 1), np.zeros(len(files) - 1),\
        np.zeros(len(files) - 1)

    # Initialize arrays for chop states
    chop_guesses, chopstate_derived = np.zeros((len(files)), dtype='<U6'), np.zeros((len(files)), dtype='<U6')
    maxima = np.zeros(((len(files)-1), 2))
    
    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        minmax, stdev, bgmean, locdiff, maxima, chop_guesses, indices = zip(*tqdm(pool.imap(ChopMetrics((files, directory, array_shape, chop_direction)), range(len(files)-1)),\
                                           desc="integrating files", total=(len(files)-1)))
        
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
    yeet = np.zeros(len(files)-1)


    # Determine chops by using recorded changes in chop state
    for i in range(len(files)-1):
        if finalcomp[i] < 1:
            chopstate_derived[i+1] = chopstate_derived[i]
        else:
            if chopstate_derived[i] == "CHOP_A":
                
                chopstate_derived[i+1] = "CHOP_B"
                yeet[i+1] = 0
                
            else:
                
                chopstate_derived[i+1] = "CHOP_A"
                yeet[i] = 1

    # Check to make sure both methods of chop determination agree
    if len(chop_guesses[chop_guesses != chopstate_derived[:-1]]) > 0:
        print(chop_guesses != chopstate_derived[:-1])        
        raise ValueError("Chop mismatch")

    chops = np.copy(chopstate_derived)

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

    # Go through groups
    for group in tqdm(chop_groups):
        frames = np.zeros((1, array_shape[0], array_shape[1]))
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
                files[group[0]+i] = ''
                chops[group[0]+i] = ''
                para_angles[group[0]+i] = 500
                
        newhdul = fits.HDUList([fits.PrimaryHDU(data=frames/count)])
        files[group[0]] = pathlib.Path(coadd_dir, files[group[0]].name)
        newhdul.writeto(files[group[0]], overwrite=True)
        newhdul.close()
        
    files = np.delete(files, np.where(np.asarray(files) == '')[0])
    chops = np.delete(chops, np.where(chops == '')[0])
    para_angles = np.delete(para_angles, np.where(para_angles == 500)[0])

    return files, chops, para_angles, chopstate_derived

def subtract_background(raw_files, psf_subtracted_files, chops, edge_cut = 2, channel_edges=[127, 255, 383], biased_columns=[303],
                        biased_rows = [], striped_regions=[], vertical_biases=[], horizontal_biases=[], nanrows=[],
                        nancols=[], flats=None, nbg=1, smooth=5, prefix="", threadcount=50):
    
    """
    Subtract background from adjacent chop frames, divide by flat, perform highpass filter

    Parameters:
    ----------------------
    raw_files: list or array 
        List of raw file paths, sorted 
    psf_subtracted_files: list or array 
        List of psf subtracted file paths, sorted 
    chops: string array
        List of chop states corresponding to the file list, entries are either "CHOP_A" or "CHOP_B"
    edge_cut: integer
        Number of pixels to remove at the edges of images
    channel_edges (optional): list
        List containing indices corresponding to channel edges
    biased_columns (optional): list
        List containing indices corresponding to biased columns
    striped_regions (optional): list
        List containing indices corresponding to regions of the image needing destriping. Each entry contains four integers for slicing the image: [0:1, 2:3]
    vertical_biases (optional): list
        List containing indices corresponding to columns separating regions of the image with different biases
    horizontal_biases (optional): list
        List containing indices corresponding to rows separating regions of the image with different biases
    biased_rows (optional): list
        List containing indices corresponding to biased rows
    nanrows (optional): list
        List containing indices corresponding to rows that need to be set to np.nan
    nancols (optional): list
        List containing indices corresponding to columns that need to be set to np.nan
    flats (optional): list
        List containing the flats for each chop state
    nbg (optional): integer
        Number of frames to use in rolling background subtraction
    smooth (optional): integer
        Radius of smoothing kernel, divided by 5
    prefix (optional): string
        Prefix to add to directory name when saving image.
    threadcount (optional): integer
        Number of threads to employ in multithreading. Default value is 50 threads.

    Returns:
    ----------------------    
    subtracted_dir: string/Path object
        Directory where subtracted images are saved
    """

    print("Subtracting backgrounds....")

    # Create directory to save files
    root_dir = os.path.dirname(os.path.dirname(raw_files[0]))
    subtracted_dir=os.path.join(root_dir, prefix+'subtracted')
    if not os.path.exists(subtracted_dir):
        os.makedirs(subtracted_dir)

    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        minima, maxima = zip(*tqdm(pool.imap(SubtractBackground((subtracted_dir, raw_files, psf_subtracted_files, chops, edge_cut, channel_edges, biased_columns,\
              striped_regions, vertical_biases, horizontal_biases, biased_rows, nanrows, nancols, flats, nbg, smooth)), range(len(raw_files))), total=len(raw_files)))

    return subtracted_dir 

def frame_registration(files, subtracted_dir, prefix='', windowsize=20, nan_mask_size=2, threadcount=50):
            
    """
    Aligns all frames together, centering the PSF in the middle of the image by translation and padding.
    
    Parameters:
    ----------------------
    files: list or array 
        List of file paths, sorted 
    subtracted_dir: string/Path object
        Directory where images are read from
    aligned_dir: string/Path object
        Directory where aligned images will be saved
    prefix (optional): string
        Prefix to add to directory name when saving image.
    windowsize (optional): integer
        Half width/height of the reference cutout image (which is 1:1 aspect ratio). Default is 20 pixels.
    nan_mask_size (optional): float
        Diameter of the mask used to mask out the oversubtracted/negative image PSF, in units of the measured Gaussian standard deviation.
    threadcount (optional): integer
        Number of threads to employ in multithreading. Default value is 50 threads.
    
    Returns:
    ----------------------    
    aligned_files: list or array
        List of aligned file paths, sorted 
    original_psf_locs: 2x len(files) numpy array
        Array containing pixel coordinates of psf locations in the input images    
    psfmaxima: float
        Measured maximum pixel value of the PSF.
    background_dev: float
        Measured standard deviation of the background.
    reffit: array
        Array containing Gaussian fit parameters:
        
        amplitude: float
            Amplitude of the Gaussian fit of the PSF.
        sigmax: float
            Standard deviation along x of the Gaussian fit of the PSF.
        sigmay: float
            Standard deviation along y of the Gaussian fit of the PSF.
        gauss_offset: float
            Offset of the Gaussian fit of the PSF.
            
    array_shape: integer tuple
        Tuple containing image dimensions, from numpy.shape
    file_size: float
        File size of aligned images.
    """
            
    print("Aligning frames....")    

    # Create directory to save files
    root_dir = os.path.dirname(subtracted_dir)
    aligned_dir = os.path.join(root_dir, prefix+'aligned')
    if not os.path.exists(aligned_dir):
        os.makedirs(aligned_dir)

    # Create meshgrid window for subpixel alignment of PSF
    wx = np.linspace(0, 2*windowsize-1, 2*windowsize)
    wy = np.linspace(0, 2*windowsize-1, 2*windowsize)
    wx, wy = np.meshgrid(wx, wy)

    # Open two frames (of different chop states)
    hdul = fits.open(os.path.join(subtracted_dir, "subtracted_"+files[0].name))
    hdul2 = fits.open(os.path.join(subtracted_dir, "subtracted_"+files[1].name))
    
    first_frame = hdul[0].data
    second_frame = hdul2[0].data

    # Find maxima and minima of both frames
    first_maxima = np.where(first_frame == np.nanmax(first_frame))
    second_maxima = np.where(second_frame == np.nanmax(second_frame))
    first_minima = np.where(first_frame == np.nanmin(first_frame))
    second_minima = np.where(second_frame == np.nanmin(second_frame))

    hdul.close()
    hdul2.close()

    # Get shape of frame
    frameh, framew = np.shape(first_frame)

    # Create x, y lists for the image
    x = np.linspace(0, framew-1, framew)
    y = np.linspace(0, frameh-1, frameh)

    # Get cutouts of PSFs in both frames
    first_cutout = first_frame[(first_maxima[0][0]-windowsize):(first_maxima[0][0]+windowsize),(first_maxima[1][0]-windowsize):(first_maxima[1][0]+windowsize)]
    second_cutout = second_frame[(second_maxima[0][0]-windowsize):(second_maxima[0][0]+windowsize),(second_maxima[1][0]-windowsize):(second_maxima[1][0]+windowsize)]
    
    # Fit an airy disk to the PSF of the first frame
    reffit, _ = curve_fit(hf.airydisk_ravel, (wx, wy), first_cutout.ravel(), p0=[np.max(first_cutout),\
                        10,10, -1, 0, windowsize, windowsize],\
                       bounds=([0, 1, 1, 1*-np.inf, 0, 1, 1], [2*np.max(first_cutout), 200, 200, np.inf, 2*np.pi, 2*windowsize, 2*windowsize]))

    # Create airy disk reference for aligning frames, but centered exactly in the middle of the cutout
    reference = hf.airydisk((wx, wy), reffit[0], reffit[1], reffit[2], reffit[3], reffit[4], windowsize-0.5, windowsize-0.5)

    # Get the offsets of the cutouts from the reference 
    first_offset = chi2_shift(reference, first_cutout, upsample_factor='auto', return_error=False)
    second_offset = chi2_shift(reference, second_cutout, upsample_factor='auto', return_error=False)

    # Include offset of second image from first image
    second_offset[0] += (second_maxima[1][0] - first_maxima[1][0])
    second_offset[1] += (second_maxima[0][0] - first_maxima[0][0])

    # Calculate necessary padding to align second image
    padding = (int(np.ceil(np.abs(second_offset[0]))*np.sign(second_offset[0])),\
               int(np.ceil(np.abs(second_offset[1]))*np.sign(second_offset[1])))

    # Create new x, y lists including padding
    px = np.linspace(0, framew-1+np.abs(padding[0]), framew+np.abs(padding[0]))
    py = np.linspace(0, frameh-1+np.abs(padding[1]), frameh+np.abs(padding[1]))

    '''
    # Set the region around the oversubtracted PSF to nans
    first_frame[hf.circular_mask((first_minima[1][0], first_minima[0][0]), nan_mask_size*reffit[1], framew, frameh)] = np.nan
    second_frame[hf.circular_mask((second_minima[1][0], second_minima[0][0]), nan_mask_size*reffit[1], framew, frameh)] = np.nan
    '''

    # Align first frame
    first_frame = hf.align_frame(first_frame, px, py, padding, first_offset)

    # Calculate the origin of the frame after alignment
    origin = [first_maxima[1][0]-0.5, first_maxima[0][0]-0.5]

    # Include the padding required for the second frame in the calculation of the origin
    if padding[1] >= 0:
        origin[1] = origin[1] + padding[1]
    if padding[0] >= 0:
        origin[0] = origin[0] + padding[0]

    # Calculate padding required to center the PSF in the frame
    center_padding = (-1*int(2*origin[0] - np.shape(first_frame)[1] + 1), -1*int(2*origin[1] - np.shape(first_frame)[0] + 1))

    # Pad first frame
    first_frame = hf.pad_frame(first_frame, len(px) + np.abs(center_padding[0]), len(py) + np.abs(center_padding[1]), center_padding)

    # Save first frame
    newhdul = fits.HDUList([fits.PrimaryHDU(data=(first_frame))])
    newhdul.writeto(os.path.join(aligned_dir, "aligned_"+files[0].name), overwrite=True)

    # Align second frame
    second_frame = hf.align_frame(second_frame, px, py, padding, second_offset)
    second_frame = hf.pad_frame(second_frame, len(px) + np.abs(center_padding[0]), len(py) + np.abs(center_padding[1]), center_padding)

    # Save second frame
    newhdul = fits.HDUList([fits.PrimaryHDU(data=(second_frame))])   
    newhdul.writeto(os.path.join(aligned_dir, "aligned_"+files[1].name), overwrite=True)
    newhdul.close()

    # Align the rest of the images
    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        offsets, _ = zip(*tqdm(pool.imap(RegisterFrames(( subtracted_dir,\
                    aligned_dir, padding, center_padding, px, py, reference, first_maxima, windowsize,\
                            nan_mask_size*reffit[1])), files[2:]), total=len(files) - 2))

    # Include first two frame offsets
    offsets = np.concatenate((np.asarray([(first_offset[0], first_offset[1]), (second_offset[0], second_offset[1])]), np.asarray(offsets)))

    # Get file paths
    aligned_files = np.asarray(sorted(list(pathlib.Path(str(aligned_dir)).rglob('*.fits'))))

    # Calculate the location of the PSF in the original unaligned images
    original_psf_locs = np.asarray([( frameh -padding[0] - center_padding[0] )/2 + offsets[:,0] - 0.5,\
                                ( framew - padding[1] - center_padding[1])/2 + offsets[:,1] - 0.5]).T
        
    return aligned_files, original_psf_locs, reffit, np.shape(first_frame), float(first_frame.nbytes)

def frame_evaluation(aligned_files, chops, array_shape, file_size, tolerance=0.9, pxscale=0.0179, windowsize=20, threadcount=50, memoryMode=1):

    """
    Evaluates all the frames, measuring the FWHM, eccentricities, maxima, background deviations, maximum cross correlation, and Gaussian fit parameters.
    
    Parameters:
    ----------------------
    aligned_files: list or array
        List of aligned file paths, sorted 
    chops: string array
        List of chop states corresponding to the file list, entries are either "CHOP_A" or "CHOP_B"
    array_shape: integer tuple
        Tuple containing image dimensions, from numpy.shape
    file_size: float
        File size of aligned images.
    tolerance (optional): float
        Fraction of available memory to be used for integration.
    pxscale (optional): float
        The image scale of the images in arcseconds per pixel. Default is 0.0179"/px, for LBTI-NOMIC.
    windowsize (optional): integer
        Half width/height of the reference cutout image (which is 1:1 aspect ratio). Default is 20 pixels.
    threadcount (optional): integer
        Number of threads to employ in multithreading. Default value is 50 threads.
    memoryMode (optional): integer
        If set to 0, frames are retrieved from aligned_files as an image cube (from memory). If set to 1, frames are opened from aligned_files as a list of files. Default is 1.
    
    Returns:
    ----------------------    
    fwhms: float array
        The measured FWHMs of the PSF, from the Gaussian fit standard deviations.
    eccentricities: float array
        The measured eccentricities of the PSF, from the Gaussian fit standard deviations.
    psfmaxima: float array
        Measured maximum pixel values of the PSF.
    background_dev: float array
        Measured standard deviations of the background.
    corr: float
        Maximum values of the cross correlation of the frames and their respective mean frames.
    amplitude: float
        Amplitude of the Gaussian fit of the PSF.
    sigmax: float
        Standard deviations along x of the Gaussian fit of the PSF.
    sigmay: float array
        Standard deviations along y of the Gaussian fit of the PSF.
    gauss_offset: float array
        Offsets of the gaussian fit of the PSF.
    """

    wx = np.linspace(0, 2*windowsize-1, 2*windowsize)
    wy = np.linspace(0, 2*windowsize-1, 2*windowsize)
    wx, wy = np.meshgrid(wx, wy)

    # Files are in memory
    if memoryMode == 0:

        # Create averaged frames for each chop state
        chopa_mean_frame = np.nanmean(aligned_files[chops == "CHOP_A"], axis=0)
        chopb_mean_frame = np.nanmean(aligned_files[chops == "CHOP_B"], axis=0)

        # Set nans to 0 allow correlation to proceed
        chopb_mean_frame[np.isnan(chopb_mean_frame)] = 0
        chopa_mean_frame[np.isnan(chopa_mean_frame)] = 0

        #if __name__ == "__main__":
        with Pool(threadcount) as pool:
            psfmaxima, background_dev, correlations, amplitudes, sigmax, sigmay, gauss_offsets =\
                zip(*tqdm(pool.imap(EvaluateFrames((aligned_files, chopa_mean_frame, chopb_mean_frame, wx, wy, windowsize, array_shape)),\
                                    np.array((chops, np.arange(len(aligned_files)))).T), total=len(aligned_files)))
        
    else:

        # Check available memory
        stats = psutil.virtual_memory()  # returns a named tuple
        available = float(getattr(stats, 'available'))

        # Separate files into chops
        chopa_files = aligned_files[chops == "CHOP_A"]
        chopb_files = aligned_files[chops == "CHOP_B"]

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
        
        #if __name__ == "__main__":
        with Pool(threadcount) as pool:
            b_bigarr, b_filecounts = zip(*tqdm(pool.imap(hf.IntegrateFrames((array_shape)), b_filebufs),\
                                               desc="integrating files", total=len(b_filebufs)))

        # Create averaged frames for each chop state
        chopa_mean_frame = np.sum(a_bigarr, axis=0)/np.sum(a_filecounts)
        
        chopb_mean_frame = np.sum(b_bigarr, axis=0)/np.sum(b_filecounts)

        chopb_mean_frame[np.isnan(chopb_mean_frame)] = 0
        chopa_mean_frame[np.isnan(chopa_mean_frame)] = 0

        #if __name__ == "__main__":
        with Pool(threadcount) as pool:
            psfmaxima, background_dev, correlations, amplitudes, sigmax, sigmay, gauss_offsets =\
                zip(*tqdm(pool.imap(EvaluateFrames((None, chopa_mean_frame, chopb_mean_frame, wx, wy, windowsize, array_shape)),\
                                    np.array((chops, aligned_files)).T), total=len(aligned_files)))

    # Convert lists into numpy arrays
    sigmax, sigmay = np.asarray(sigmax), np.asarray(sigmay)

    # Calculate fwhms
    fwhms = 2*np.sqrt(2*np.log(2)*(sigmax+sigmay))*pxscale

    # Calculate eccentricities, handling the potential imaginary radical
    eccentricities = np.sqrt(1 - sigmax/sigmay)
    eccentricities_r = np.sqrt(1 - sigmay/sigmax)
    eccentricities[np.isnan(eccentricities)] = eccentricities_r[np.isnan(eccentricities)]
    
    return fwhms, eccentricities, np.asarray(psfmaxima), np.asarray(background_dev), np.asarray(correlations), np.asarray(amplitudes), np.asarray(gauss_offsets), sigmax, sigmay

def frame_rejection(chops, params, sigma=None):

    """
    Create a badmap by constructing a crude flat, running it through a high pass filter, and masking out large deviations.
    
    Parameters:
    ----------------------
    chops: string array
        List of chop states corresponding to the file list, entries are either "CHOP_A" or "CHOP_B"
    tolerance (optional): float
        Fraction of available memory to be used for integration.
    sigma (optional): float
        Number of standard deviations to include in the badmap.
    edge_cut (optional): integer
        Number of pixels to remove from the edges of the image before high pass filtering. Default value is 3 pixels.
    threadcount (optional): integer
        Number of threads to employ in multithreading. Default value is 50 threads.
    
    Returns: 
    ---------------------- 
    bools: boolean array
        Boolean mask array denoting which frames are rejected (rejected frame indices are set to False)
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
        params[i][chopa_bool] += np.nanmedian(params[i][chopb_bool]) - np.nanmedian(params[i][chopa_bool])

    # Initialize boolean array by removing nans in first parameter
    bools = ~np.isnan(params[0])

    # Iterate over parameters to reject frames
    for i in range(len(params)):
        
        bools = bools & (params[i] < np.nanmedian(params[i]) + sigma[i]*np.nanstd(params[i]))

    return bools
    
# Deprecated
def fractional_frame_rejection(psfmaxima, background_dev, fwhms, eccentricities, correlations, amplitudes, gauss_offsets,\
                               fraction_frames=0.3, start_sigma=5, fev=100):

    sigma = start_sigma+0.1
    frac_frame_bool = 1
    count = 0

    while (count < fev):
        sigma -= 0.1
        frame_bool = frame_rejection(psfmaxima, background_dev, fwhms, eccentricities, correlations, amplitudes, gauss_offsets, sigma=sigma)
        frac_frame_bool = len(frame_bool[frame_bool == False])/len(frame_bool)
        if (frac_frame_bool >= fraction_frames):
            return frame_bool, sigma
        count += 1
    raise ValueError("Exceeded number of iterations!")


def frame_binning(aligned_files, frame_bool, chops, para_angles, array_shape, prefix='', bin=50, threadcount=50, memoryMode=1):

    """
    Bin frames temporally, and update chop states and parallactic angle.
    
    Parameters:
    ----------------------
    aligned_files: list or array 
        List of aligned file paths, sorted.
    frame_bool: boolean array
        Boolean mask array denoting which frames to exclude from binning.
    chops: string array
        List of chop states corresponding to the file list, entries are either "CHOP_A" or "CHOP_B"
    para_angles: list or array
        List of parallactic angles corresponding to each file in 'aligned_files'
    array_shape: integer tuple
        Tuple containing image dimensions, from numpy.shape
    prefix (optional): string
        Prefix to add to directory name when saving image.
    bin (optional): integer
        Number of frames to add for each binned frame. Default value is 50 frames.
    threadcount (optional): integer
        Number of threads to employ in multithreading. Default value is 50 threads.
    memoryMode (optional): integer
        If set to 0, frames are returned to memory. If set to 1, frames are saved to files. Default is 1.
    
    Returns: 
    ---------------------- 
    binned_files: Path array or array of 2D images
        If memoryMode is 0, this variable contains all of the binned frames as an image cube.
        Otherwise, this variable is a array of file paths to the binned frames.
    binned_chops: string array
        List of chop states corresponding to each image in 'binned_files', entries are either "CHOP_A" or "CHOP_B"
    binned_angles: array
        List of parallactic angles corresponding to each image in 'binned_files'
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
            a_binned_frames, a_binned_angles, a_binned_filenames  = zip(*tqdm(pool.imap(BinFrames((array_shape, None)), zip(a_angles, a_binfiles))))

        #if __name__ == "__main__":
        with Pool(threadcount) as pool:
            b_binned_frames, b_binned_angles, b_binned_filenames = zip(*tqdm(pool.imap(BinFrames((array_shape, None)), zip(b_angles, b_binfiles))))

        # Sort filenames in order
        binned_filenames = np.asarray(sorted(a_binned_filenames+b_binned_filenames))

        # Convert into numpy arrays
        a_binned_frames, a_binned_angles, a_binned_filenames = np.asarray(a_binned_frames), np.asarray(a_binned_angles), np.asarray(a_binned_filenames)
        b_binned_frames, b_binned_angles, b_binned_filenames = np.asarray(b_binned_frames), np.asarray(b_binned_angles), np.asarray(b_binned_filenames)
        

        # Populate chop list with chop states
        binned_chops[np.where(np.isin(binned_filenames,a_binned_filenames) == True)[0]] = "CHOP_A"
        binned_chops[np.where(np.isin(binned_filenames,b_binned_filenames) == True)[0]] = "CHOP_B"

        # Populate binned_angles and binned_frames with angles and frames in order
        for i in range(numbinfiles):

            if binned_filenames[i] in a_binned_filenames:
                
                binned_angles[i] = a_binned_angles[np.where(a_binned_filenames == binned_filenames[i])[0]]
                binned_frames[i] = a_binned_frames[np.where(a_binned_filenames == binned_filenames[i])[0]]
                
            else:
                
                binned_angles[i] = b_binned_angles[np.where(b_binned_filenames == binned_filenames[i])[0]]
                binned_frames[i] = b_binned_frames[np.where(b_binned_filenames == binned_filenames[i])[0]]
        
        return binned_frames, binned_chops, binned_angles

    # Save frames to files and return paths
    else:
            
        #if __name__ == "__main__":
        with Pool(threadcount) as pool:
            a_binned_angles, a_binned_filenames  = zip(*tqdm(pool.imap(BinFrames((array_shape, binned_dir)), zip(a_angles, a_binfiles))))
        a_binned_angles, a_binned_filenames = np.asarray(a_binned_angles), np.asarray(a_binned_filenames)
        
        #if __name__ == "__main__":
        with Pool(threadcount) as pool:
            b_binned_angles, b_binned_filenames = zip(*tqdm(pool.imap(BinFrames((array_shape, binned_dir)), zip(b_angles, b_binfiles))))
        b_binned_angles, b_binned_filenames = np.asarray(b_binned_angles), np.asarray(b_binned_filenames)


        # Create file list by reading path
        binned_files = np.asarray(sorted(list(pathlib.Path(str(binned_dir)).rglob('*.fits'))))

        # Get filenames
        binned_filenames = np.asarray([f.name for f in binned_files])

        # Populate chop list with chop states
        binned_chops[np.where(np.isin(binned_filenames,a_binned_filenames) == True)[0]] = "CHOP_A"
        binned_chops[np.where(np.isin(binned_filenames,b_binned_filenames) == True)[0]] = "CHOP_B"

        # Populate binned_angles and binned_frames with angles and frames in order
        for i in range(len(binned_files)):
            
            if binned_filenames[i] in a_binned_filenames:
                
                binned_angles[i] = a_binned_angles[np.where(a_binned_filenames == binned_filenames[i])[0]]
                
            else:
                
                binned_angles[i] = b_binned_angles[np.where(b_binned_filenames == binned_filenames[i])[0]]
        
        return binned_files, binned_chops, binned_angles

