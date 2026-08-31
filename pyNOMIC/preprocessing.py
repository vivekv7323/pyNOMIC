#----------------------------------------
# IMPORTS
#----------------------------------------
import os, pathlib
import numpy as np
from tqdm.auto import tqdm
from multiprocessing.pool import ThreadPool as Pool

from astropy.io import fits
from astropy.convolution import (convolve_fft, Box2DKernel,
                                 Ring2DKernel, Gaussian2DKernel)

from scipy.optimize import curve_fit
from scipy.ndimage import median_filter
from scipy.interpolate import NearestNDInterpolator

from image_registration import chi2_shift

import pyNOMIC.helper_functions as hf

#----------------------------------------
# CLASSES
#----------------------------------------

class PSFSubtraction(object):

    """
    Subtracts stellar psfs from raw images.
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
            List of chop states corresponding to the file list, entries
            are either "CHOP_A" or "CHOP_B"
        maxima: float tuple array
            Tuples encoding location of the PSF in the images
        badmap: 2D image array
            Bad pixel map, where bad pixels are set to 0 and all other
            pixels are set to 1.
        starmasks: List of 2D boolean arrays
            Star masks applied to images for detecting the star,
            affected pixels are set to np.nan
        flats: List of 2D numpy arrays
            Temporary flat applied to locate the star
        wvl_interp: 1D numpy array
            An array of wavelengths. 
        relative_flux: 1D numpy array
            Relative flux for each respective wavelength in wvl_interp.
        windowsize: integer
            Half width/height of the cutout image (which is 1:1 aspect
            ratio)
        nbg: integer
            Number of frames to use in rolling background subtraction
        smooth: integer
            Radius of smoothing kernel, divided by 5
        recur_iteration (optional): integer
            Number of recursive iterations allowed in psf fitting,
            default is 2.
        remove_trefoil: boolean
            Enables removal of psf residual from trefoil through
            highpass filtering.
        remove_residual: boolean
            Enables removal of psf residual through highpass filtering.
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
        maximum[1]: float
            Y-axis location of the maximum
        failcode: integer
            Integer encoding the success of each PSF fit.
        reffit: 1D numpy array
            Airy disk fitting parameters
        lbtfit: 1D numpy array
            Empirical PSF fitting parameters
        trifit: 1D numpy array
            Trefoil PSF fitting parameters
        """
        
        (psf_subtracted_dir, files, chops, maxima, badmap, starmasks,
         flats, wvl_interp, relative_flux, windowsize, nbg, smooth, 
         recur_iteration, remove_trefoil, remove_residual) = self.params

        # Open image and get array shape
        unsubtracted = fits.open(files[i])
        img = unsubtracted[0].data
        if img.ndim > 2:
            img = img[0]
        array_shape = np.shape(img)

        subtracted_frame = hf.chop_subtraction(img, i, chops[i], files, flats, nbg)    

        if badmap is not None:
            # Create boolean map from badmap to remove bad pixels
            subtracted_frame[badmap < 1] = np.nan
        
        # Subtract convolved background
        bg_model = convolve_fft(np.pad(subtracted_frame, 10*smooth, mode='edge'),
                                Ring2DKernel(5*smooth, 4*smooth))[10*smooth:-10*smooth,
                                                                  10*smooth:-10*smooth]
        
        # Create model grid
        nx = np.linspace(0, array_shape[1]-1, array_shape[1])
        ny = np.linspace(0, array_shape[0]-1, array_shape[0])
        nx, ny = np.meshgrid(nx, ny)

        if maxima is None:
            test_frame = subtracted_frame - bg_model
            test_frame = convolve_fft(test_frame, Box2DKernel(smooth*3))
            if starmasks is not None:
                if chops[i] == "CHOP_A":
                    test_frame[(badmap == 0) | (starmasks[0] == 0)] = np.nan
                else:
                    test_frame[(badmap == 0) | (starmasks[1] == 0)] = np.nan
            else:
                raise ValueError("starmask is not provided")
            maximum = np.asarray(np.where((test_frame) == np.nanmax(test_frame)))[:,0]
        else:
            maximum = maxima[i]

        for j in range(recur_iteration):

            # Perform background subtraction
            new_frame = subtracted_frame - bg_model

            # Create a cutout of the frame
            cutout = new_frame[(int(maximum[0])-windowsize):(int(maximum[0])+windowsize),
                               (int(maximum[1])-windowsize):(int(maximum[1])+windowsize)]

            # Remove nans via interpolation
            mask = np.where(~np.isnan(cutout))
            interp = NearestNDInterpolator(np.transpose(mask), cutout[mask])
            cutout = interp(*np.indices(cutout.shape))

            # Variable to keep track of fitting failures
            failcode = 0

            # Fit cutout to get empirical psf parameters
            reffit, lbtfit, trifit = hf.empirical_psf_fit(cutout, wvl_interp, relative_flux,
                                                          model_trefoil=remove_trefoil)
            if not np.isnan(reffit[0]):
                # Get origin of the PSF in the frame
                origin = (int(maximum[1])-windowsize+reffit[5],
                          int(maximum[0])-windowsize+reffit[6])

                if not np.isnan(lbtfit[0]):
                    # Create psf model by integrating over wavelength
                    psf_model = lbtfit[0]*np.mean(hf.modified_airy_disk((nx, ny), relative_flux,
                                                                        wvl_interp, lbtfit[2], lbtfit[3],
                                                                        0, reffit[4], origin[0],
                                                                        origin[1]),
                                                  axis=0)/np.mean(relative_flux) + lbtfit[1]
                else:
                    # Create psf model with airy disk
                    psf_model = hf.airy_disk((nx, ny), reffit[0], reffit[1], reffit[2], reffit[3],
                                             reffit[4], reffit[5], reffit[6], ravel=False)
                    failcode += 10
                
                if remove_trefoil and ~np.isnan(trifit[0]):
                    # Add trefoil model
                    psf_model += hf.center_triangle((nx, ny), trifit[0], trifit[1], trifit[2],
                                                    trifit[3], trifit[4], origin[0], origin[1],
                                                    trifit[5], trifit[6], ravel=False)
    
                # Create background model by subtracting the psf model and convolving with ring kernel
                bg_model = convolve_fft(subtracted_frame - psf_model, Ring2DKernel(3*smooth, 2*smooth))
            else:
                return np.nan, np.nan, 111, reffit, lbtfit, trifit

        # Keep track of trifit failure if remove_trefoil is enabled
        if remove_trefoil and np.isnan(trifit[0]):
            failcode += 1
        
        if remove_residual:

            # Smoothen out residual
            residual = (subtracted_frame - psf_model)
            residual_bg = convolve_fft(residual, Gaussian2DKernel(2))

            # Create residual removal mask
            radius = 1.2*int(np.sqrt(reffit[1]**2 + reffit[2]**2))
            psfrem = hf.psf_removal_mask(origin, radius, 1.2*radius, array_shape[0],
                                         array_shape[1])

            # Create circular sampling mask
            cirmask = (hf.circular_mask(origin, radius, array_shape[0], array_shape[1]) ^
                       hf.circular_mask(origin, 1.2*radius, array_shape[0], array_shape[1]))

            # Multiple flat back
            if flats is not None:
                if chops[i] == "CHOP_A":
                    psf_model = psf_model*flats[0]/np.nanmedian(flats[0])
                else:
                    psf_model = psf_model*flats[1]/np.nanmedian(flats[1])

            # Remove residual
            final = (img - psf_model)*(1-psfrem) + (img - psf_model - residual_bg +
                                                    np.median(residual_bg[cirmask]))*psfrem
        else:
            if flats is not None:
                if chops[i] == "CHOP_A":
                    psf_model = psf_model*flats[0]/np.nanmedian(flats[0])
                else:
                    psf_model = psf_model*flats[1]/np.nanmedian(flats[1])
            final = img - psf_model

        unsubtracted.close()
        
        # Write image to file
        newhdul = fits.HDUList([fits.PrimaryHDU(data=(final))])
        newhdul.writeto(os.path.join(psf_subtracted_dir, "psfsubtracted_"+files[i].name),
                        overwrite=True)
        newhdul.close()

        return origin[1], origin[0], failcode, reffit, lbtfit, trifit


class PSFSubRedux(object):

    """
    Subtracts stellar psfs from raw images with failed PSF fits.
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
            List of chop states corresponding to the file list, entries
            are either "CHOP_A" or "CHOP_B"
        maxima: float tuple array
            Tuples encoding location of the PSF in the images
        failcodes: integer array
            Integers encoding the success of each PSF fit corresponding
            to the file list.
        reffits: 2D numpy array
            Airy disk fitting parameters corresponding to each file.
        lbtfits: 2D numpy array
            Empirical PSF fitting parameters corresponding to each file.
        trifits: 2D numpy array
            Trefoil fitting parameters corresponding to each file.
        badmap: 2D image array
            Bad pixel map, where bad pixels are set to 0 and all other
            pixels are set to 1.
        flats: List of 2D numpy arrays
            Temporary flats applied to locate the star
        wvl_interp: 1D numpy array
            An array of wavelengths. 
        relative_flux: 1D numpy array
            Relative flux for each respective wavelength in wvl_interp.
        windowsize: integer
            Half width/height of the cutout image (which is 1:1 aspect
            ratio)
        fit_reject_criterion (optional): integer
            Maximum allowed failcode, used in replacing failed psf fits.
        nbg: integer
            Number of frames to use in rolling background subtraction
        smooth: integer
            Radius of smoothing kernel, divided by 5
        remove_residual: boolean
            Enables removal of psf residual through highpass filtering.
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
        maximum[1]: float
            Y-axis location of the maximum
        """
        
        (psf_subtracted_dir, files, chops, maxima, failcodes,
         reffits, lbtfits, trifits, badmap, flats, wvl_interp,
         relative_flux, windowsize, fit_reject_criterion,
         nbg, smooth, remove_residual) = self.params

        # Open image and get array shape
        unsubtracted = fits.open(files[i])
        img = unsubtracted[0].data
        if img.ndim > 2:
            img = img[0]
        array_shape = np.shape(img)

        # Create model grid
        nx = np.linspace(0, array_shape[1]-1, array_shape[1])
        ny = np.linspace(0, array_shape[0]-1, array_shape[0])
        nx, ny = np.meshgrid(nx, ny)

        # Find adjacent fits and maxima and initialize here
        file_indices = np.arange(len(files))[failcodes <= fit_reject_criterion]
        dist = np.abs(file_indices - i)
        fit_indices = file_indices[np.where(dist == np.min(dist))[0]]

        maximum = np.mean(maxima[fit_indices], axis=0)
        reffit = np.mean(reffits[fit_indices], axis=0)
        
        # Get origin of the PSF in the frame
        origin = (int(maximum[1])-windowsize+reffit[5],
                  int(maximum[0])-windowsize+reffit[6])

        if (np.isin(0, failcodes[fit_indices]) |\
            np.isin(1, failcodes[fit_indices])):

            lbtfit = np.nanmean(lbtfits[fit_indices], axis=0)
            # Create psf model by integrating over wavelength
            psf_model = lbtfit[0]*np.mean(hf.modified_airy_disk((nx, ny), relative_flux,
                                                                wvl_interp, lbtfit[2], lbtfit[3],
                                                                0, reffit[4], origin[0],
                                                                origin[1]),
                                          axis=0) + lbtfit[1]
        else:
            # Create psf model with airy disk
            psf_model = hf.airy_disk((nx, ny), reffit[0], reffit[1], reffit[2], reffit[3],
                                     reffit[4], reffit[5], reffit[6], ravel=False)
        
        if np.isin(False, np.isnan(trifits[fit_indices])):
            
            trifit = np.nanmean(trifits[fit_indices], axis=0)
            # Add trefoil model
            psf_model += hf.center_triangle((nx, ny), trifit[0], trifit[1], trifit[2],
                                            trifit[3], trifit[4], origin[0], origin[1],
                                            trifit[5], trifit[6], ravel=False)

        if remove_residual:

            subtracted_frame = hf.chop_subtraction(img, i, chops[i], files, flats, nbg)    
    
            if badmap is not None:
                # Create boolean map from badmap to remove bad pixels
                subtracted_frame[badmap < 1] = np.nan

            # Smoothen out residual
            residual = (subtracted_frame - psf_model)
            residual_bg = convolve_fft(residual, Gaussian2DKernel(2))

            # Create residual removal mask
            radius = 1.2*int(np.sqrt(reffit[1]**2 + reffit[2]**2))
            psfrem = hf.psf_removal_mask(origin, radius, 1.2*radius, array_shape[0],
                                         array_shape[1])

            # Create circular sampling mask
            cirmask = (hf.circular_mask(origin, radius, array_shape[0], array_shape[1]) ^
                       hf.circular_mask(origin, 1.2*radius, array_shape[0], array_shape[1]))

            if flats is not None:
                if chops[i] == "CHOP_A":
                    psf_model = psf_model*flats[0]/np.nanmedian(flats[0])
                else:
                    psf_model = psf_model*flats[1]/np.nanmedian(flats[1])

            # Remove residual
            final = (img - psf_model)*(1-psfrem) + (img - psf_model - residual_bg +
                                                    np.median(residual_bg[cirmask]))*psfrem
        else:
            
            if flats is not None:
                if chops[i] == "CHOP_A":
                    psf_model = psf_model*flats[0]/np.nanmedian(flats[0])
                else:
                    psf_model = psf_model*flats[1]/np.nanmedian(flats[1])
            final = img - psf_model

        unsubtracted.close()
        
        # Write image to file
        newhdul = fits.HDUList([fits.PrimaryHDU(data=(final))])
        newhdul.writeto(os.path.join(psf_subtracted_dir, "psfsubtracted_"+files[i].name),
                        overwrite=True)
        newhdul.close()

        return origin[1], origin[0]

class ChopSubtract(object):
    
    """
    Runs pyNOMIC.helper_functions.chop_subtraction
    through parallelization.
    """
    
    def __init__(self, params):

        """
        Parameters:
        ----------------------
        files: list or array 
            List of file paths, sorted 
        chops: string array
            List of chop states corresponding to the file list, entries
            are either "CHOP_A" or "CHOP_B"
        directory: string/Path object
            Directory where images will be saved
        """

        self.params = params

    def __call__(self, i):

        """
        Parameters:
        ----------------------
        index: integer
            File index to process from 'files'
        """

        (files, chops, directory) = self.params

        hdul = fits.open(files[i])
        img = hdul[0].data
        hdul.close()

        subtracted_frame = hf.chop_subtraction(img, i, chops[i], files, [0,0], 1,
                                               correction_method="subtraction")
        
        # Write image to file
        newhdul = fits.HDUList([fits.PrimaryHDU(data=(subtracted_frame))])
        newhdul.writeto(os.path.join(directory, "chopsub_"+files[i].name),
                        overwrite=True)
        newhdul.close()

        return True, True

class ChopAlign(object):

    def __init__(self, params):
        
        """
        Parameters:
        ----------------------
        files: list or array 
            List of file paths, sorted 
        chops: string array
            List of chop states corresponding to the file list, entries
            are either "CHOP_A" or "CHOP_B"
        directory: string/Path object
            Directory where images will be saved
        reference: 2D numpy array
            Reference image for FFT alignment
        px: 1D numpy array
            Array enumerating columns of the alignment grid
        py: 1D numpy array
            Array enumerating rows of the alignment grid
        ref_index: integer
            File index of the reference
        smooth: integer
            Radius of smoothing kernel
        channel_edges: list of integers
            List containing indices corresponding to the central row of
            channel edges
        interp_method: string
            Interpolation method for
            scipy.interp.RegularGridInterpolator,
            default is cubic interpolation. Linear interpolation is much
            faster but imprecise especially at the center of the PSF.
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
        offset[0]: float
            X-axis offset between reference and image
        offset[1]: float
            Y-axis offset between reference and image
        chopres_std: float
            Standard deviation of the chop residual map
        highfreq_std: float
            Standard deviation of the high frequency component
        """
        
        (files, chops, chopres_dir, highfreq_dir, reference, px, py,
         ref_index, smooth, channel_edge, interp_method) = self.params

        hdul = fits.open(files[i])
        img = hdul[0].data
        hdul.close()

        # Chop subtraction
        subtracted_frame = hf.chop_subtraction(img, i, chops[i], files, [0,0], 1,
                                               correction_method="subtraction")

        # Remove channel edges with interpolation
        for channel_edge in channel_edges:
            repaired_frame = hf.repair_channel_edges(subtracted_frame, channel_edge)

        # Convolve with a Gaussian kernel to remove high frequencies
        convolved_frame = convolve_fft(np.pad(repaired_frame, 10*smooth, mode='edge'),
                                       Gaussian2DKernel(smooth))[10*smooth:-10*smooth,
                                                                 10*smooth:-10*smooth]

        # Preserve high frequency information
        highfreq = img - convolved_frame

        # Do FFT registration to find offsets
        if chops[ref_index] == chops[i]:
            offset = chi2_shift(reference, convolved_frame, upsample_factor='auto',
                                return_error=False)
        else:
            offset = chi2_shift(-1*reference, convolved_frame, upsample_factor='auto',
                                return_error=False)

        # Use offsets to align the image
        aligned_img = hf.align_frame(convolved_frame, px, py, (0,0), offset, method=interp_method)
        
        # Write image to file
        newhdul = fits.HDUList([fits.PrimaryHDU(data=(aligned_img))])
        newhdul.writeto(os.path.join(chopres_dir, "chopres_"+files[i].name),
                        overwrite=True)
        newhdul.close()

        # Write image to file
        newhdul = fits.HDUList([fits.PrimaryHDU(data=(highfreq))])
        newhdul.writeto(os.path.join(highfreq_dir, "highfreq_"+files[i].name),
                        overwrite=True)
        newhdul.close()

        return offset[0], offset[1], np.nanstd(aligned_img), np.nanstd(highfreq)

class SubtractBackground(object):

    '''
    Subtract background from adjacent chop frames, divide by flat,
    correct cosmetic defects, perform highpass filter
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
            List of chop states corresponding to the file list,
            entries are either "CHOP_A" or "CHOP_B"
        channel_edges: list
            List containing indices corresponding to the central row of
            channel edges
        biased_columns: list
            List containing indices corresponding to biased columns
        striped_regions: list
            List containing indices corresponding to regions of
            the image needing destriping. Each entry contains four
            integers for slicing the image: [0:1, 2:3]
        vertical_biases: list
            List containing indices corresponding to columns separating
            regions of the image with different biases
        horizontal_biases: list
            List containing indices corresponding to rows separating
            regions of the image with different biases
        biased_rows: list
            List containing indices corresponding to biased rows
        nanrows: list
            List containing indices corresponding to rows that need
            to be set to np.nan
        nancols: list
            List containing indices corresponding to columns that need
            to be set to np.nan
        flats: list
            List containing the flats for each chop state
        resflats: list
            List containing low frequency flats for correcting chop 
            residuals for each chop state.
        flat_offsets: 2D numpy array
            List of tuples containing the offsets of the background
            with respect to the flat.
        correction_method: string
            Method by which to apply background model correction.
            Options are either "subtraction" or "division",
            "division" is the default.
        channel_method: string
            Method to correct channel edge. Default is linear
            interpolation, other options include nearest neighbor
            interpolation and gradient modeling.
        nbg: integer
            Number of frames to use in rolling background subtraction
        smooth: integer
            Radius of smoothing kernel, divided by 5
        edge_cut: integer
            Number of pixels to remove at the edges of images
        """
        
        self.params = params
        
    def __call__(self, i):

        """
        Parameters:
        ----------------------
        index: integer
            File index to process from 'files'
        """     

        (subtracted_dir, raw_files, psf_subtracted_files, chops, channel_edges,
         biased_columns, striped_regions, vertical_biases, horizontal_biases,
         biased_rows, nanrows, nancols, flats, resflats, flat_offsets,
         correction_method, channel_method, nbg, smooth, edge_cut) = self.params

        # Open image and get array shape
        unsubtracted = fits.open(raw_files[i])
        psf_subtracted = fits.open(psf_subtracted_files[i])
        unsubtracted_img = unsubtracted[0].data
        if unsubtracted_img.ndim > 2:
            unsubtracted_img = unsubtracted_img[0]

        subtracted_frame = hf.chop_subtraction(unsubtracted_img, i, chops[i],
                                               psf_subtracted_files, flats, nbg,
                                               resflats=resflats, flat_offsets=flat_offsets,
                                               correction_method=correction_method)

        psf_subtracted_frame = hf.chop_subtraction(psf_subtracted[0].data, i, chops[i],
                                                   psf_subtracted_files, flats, nbg,
                                                   resflats=resflats, flat_offsets=flat_offsets,
                                                   correction_method=correction_method)
                    
        # Zero point the frames
        subtracted_frame -= np.nanmedian(subtracted_frame)
        psf_subtracted_frame -= np.nanmedian(psf_subtracted_frame)

        chopsub_stds = hf.channel_stats(psf_subtracted_frame)

        # These steps are done in a very particular order
        # Remove channel edges
        for channel_edge in channel_edges:
            
            psf_subtracted_frame[channel_edge-1:channel_edge+2, :] = np.nan

        # Remove vertical lines
        for biased_column in biased_columns:
            subtracted_frame, psf_subtracted_frame = (
                hf.repair_vertical_line(subtracted_frame, biased_column, 
                                        ref=psf_subtracted_frame, return_ref=True)
            )

        # Destripe
        for striped_region in striped_regions:
            diff = hf.destriping(psf_subtracted_frame[striped_region[0]:striped_region[1],
                                                      striped_region[2]:striped_region[3]],
                                 ref_row=striped_region[4])
            subtracted_frame[striped_region[0]:striped_region[1],
                             striped_region[2]:striped_region[3]] -= diff
            psf_subtracted_frame[striped_region[0]:striped_region[1],
                                 striped_region[2]:striped_region[3]] -= diff

        # Remove vertical biases
        for vertical_bias in vertical_biases:
            subtracted_frame, psf_subtracted_frame = (
                hf.repair_vertical_bias(subtracted_frame, vertical_bias,
                                        ref=psf_subtracted_frame, return_ref=True)
            )

        # Remove horizontal biases
        for horizontal_bias in horizontal_biases:
            subtracted_frame, psf_subtracted_frame = (
                hf.repair_horizontal_bias(subtracted_frame, horizontal_bias,
                                          ref=psf_subtracted_frame, return_ref=True)
            )

        # Remove horizontal lines
        for biased_row in biased_rows:
            subtracted_frame, psf_subtracted_frame = (
                hf.repair_horizontal_line(subtracted_frame, biased_row,
                                          ref=psf_subtracted_frame, return_ref=True)
            )

        for channel_edge in channel_edges:
            subtracted_frame = hf.repair_channel_edges(subtracted_frame,
                                                       channel_edge, method=channel_method)
            psf_subtracted_frame = hf.repair_channel_edges(psf_subtracted_frame,
                                                           channel_edge, method="linear")
    
        # Set rows and columns to nan if needed
        for loc in nanrows:
            subtracted_frame[loc, :] = np.nan
        for loc in nancols:
            subtracted_frame[:, loc] = np.nan

        # Remove image edges as they are often bad columns/rows
        subtracted_frame = subtracted_frame[edge_cut:-1*edge_cut ,edge_cut :-1*edge_cut]
        psf_subtracted_frame = psf_subtracted_frame[edge_cut:-1*edge_cut ,edge_cut :-1*edge_cut]

        # Subtract convolved background

        convolve_bg = convolve_fft(np.pad(psf_subtracted_frame, 50, mode='edge'),
                                                           Ring2DKernel(5*smooth, 4*smooth))\
                                                               [50:-50, 50:-50]

        subtracted_frame -= convolve_bg
        
        backsub_stds = hf.channel_stats(psf_subtracted_frame - convolve_bg)

        # Write image to file
        newhdul = fits.HDUList([fits.PrimaryHDU(data=(subtracted_frame))])
        newhdul.writeto(os.path.join(subtracted_dir, "subtracted_"+raw_files[i].name),
                        overwrite=True)
        newhdul.close()
    
        unsubtracted.close()
        psf_subtracted.close()

        return chopsub_stds, backsub_stds

class RegisterFrames(object):

    """
    Aligns frames using both a PSF reference and a provided PSF
    location, then pads the image as necessary.
    """

    def __init__(self, params):

        """
        Parameters (contained inside a tuple):
        ----------------------
        files: list or array 
            List of file paths, sorted 
        subtracted_dir: string/Path object
            Directory where images are read from
        aligned_dir: string/Path object
            Directory where aligned images will be saved
        padding: integer tuple
            Tuple describing number of nan columns and rows to add to
            the image when aligning
        center_padding: integer tuple
            Tuple describing number of nan columns and rows to add to
            the image to place the PSF at the center of the image
        px: 1D numpy array
            Array enumerating columns of the alignment grid
        py: 1D numpy array
            Array enumerating rows of the alignment grid
        wx: 1D numpy array
            Array enumerating columns of the alignment grid
        wy: 1D numpy array
            Array enumerating rows of the alignment grid
        reference: 2D numpy array
            Reference image cutout for subpixel PSF alignment
        first_maximum: float tuple
            Tuple encoding location of the PSF in the original
            reference image (the first image of the cube)
        windowsize: integer
            Half width/height of the reference cutout image
            (which is 1:1 aspect ratio)
        alignment_method: string
            Method by which to align PSFS, either through FFT transform
            ("fft"), or through airy disk fitting ("fitting"). Default
            is FFT.
        interp_method: string
            Interpolation method for
            scipy.interp.RegularGridInterpolator,
            default is cubic interpolation. Linear interpolation is much
            faster but imprecise especially at the center of the PSF.
        actmax: boolean
            Skips subpixel PSF measurement and instead accepts the positions
            provided by the variable "maxima" as the true location. Disabled
            by default.
        boxcar: boolean
            Enables 2x2 boxcar smoothing to combat odd/even detector
            noise. False by default.
        save_files: boolean
            Enables saving the aligned files, True by default.
            
        Returns:
        ----------------------    
        offsets: tuple
            Tuple containing the total offset of the PSF from
            the original reference image.
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
        offsets: tuple
            Tuple containing the total offset of the PSF from
            the original reference image.
        imgfit: 1D numpy array
            Airy disk fit parameters if airy disk fitting is enabled.
        """

        (files, subtracted_dir, aligned_dir, badmap, starmask, padding, center_padding, px, py,
         wx, wy, reference, first_maximum, maxima, windowsize, alignment_method, interp_method,
         actmax, boxcar, save_files) = self.params

        # Open file
        hdul = fits.open(os.path.join(subtracted_dir, "subtracted_"+files[i].name))
        frame = hdul[0].data
        array_shape = np.shape(frame)
        hdul.close()

        if maxima is None:
            
            test_frame = np.copy(frame)
            
            if badmap is not None:
                test_frame[(badmap == 0)] = np.nan
                
            if starmask is not None:
                test_frame[(starmask == 0)] = np.nan

            # Locate PSF by finding the maximum and the oversubtracted PSF by finding the minimum
            maximum = np.asarray(np.where(test_frame == np.nanmax(test_frame)))[:,0]
            
        else:
            maximum = maxima[i]

        bound = (int(maximum[0])+windowsize) - array_shape[0]
        if (bound < 0) & (bound > (2*windowsize - array_shape[0])):
            bound = 0
        elif (bound <= (2*windowsize  - array_shape[0])):
            bound = int(maximum[0]) - windowsize
        maximum[0] -= bound

        # Create cutout of PSF to save resources
        cutout = frame[(int(maximum[0])-windowsize):(int(maximum[0])+windowsize),
                       (int(maximum[1])-windowsize):(int(maximum[1])+windowsize)]

        if actmax is False:
            
            # Find offset of PSF relative to maximum
            if alignment_method == "fitting":
                try:
                    # Run curve_fit to get airy best fit parameters
                    imgfit, _ = curve_fit(hf.airy_disk, (wx, wy), cutout.ravel(),
                                          p0=[np.max(cutout), 30, 30, -1, 0, windowsize-0.5,
                                              windowsize-0.5+bound],
                                          bounds=([0, 1, 1, 1*-np.inf, 0, 1, 1],
                                                  [10*np.max(cutout), 200, 200, np.inf, 2*np.pi,
                                                   2*windowsize, 2*windowsize]))
                    offset = [imgfit[5] - (windowsize - 0.5), imgfit[6] - (windowsize - 0.5)]
                    
                except:
                    offset = chi2_shift(reference, cutout, upsample_factor='auto',
                                        return_error=False)
                    imgfit = np.full(7, np.nan)               
                    
            elif alignment_method == "fft":
                
                    offset = chi2_shift(reference, cutout, upsample_factor='auto',
                                        return_error=False)
                    imgfit = np.full(7, np.nan)
            else:
                raise ValueError("Incorrect alignment method") 
        else:
            imgfit = np.full(7, np.nan)
            offset = np.asarray([maxima[i][1] - int(maxima[i][1]),
                                 maxima[i][0] - int(maxima[i][0])])

        # Add offset of maximum from the first image to compute total offset
        offset[0] += (int(maximum[1]) - int(first_maximum[1]))
        offset[1] += (int(maximum[0]) - int(first_maximum[0]))
        
        if i % 2 == 0:
            padding = -1*padding
            offset += padding

        if save_files:
        
            frame = hf.align_frame(frame, px, py, padding, offset, method=interp_method)
            frame = hf.pad_frame(frame, len(px) + np.abs(center_padding[0]),
                                 len(py) + np.abs(center_padding[1]), center_padding)
    
            if boxcar:
                frame = convolve_fft(frame, Box2DKernel(2), preserve_nan=True)
    
            # Write image to file
            newhdul = fits.HDUList([fits.PrimaryHDU(data=(frame))])
            newhdul.writeto(os.path.join(aligned_dir, "aligned_"+files[i].name), overwrite=True)
            newhdul.close()

        return (offset[0], offset[1]), imgfit
        
#----------------------------------------
# FUNCTIONS
#----------------------------------------

def create_stacked_flat(files, chops, chop_direction="UP-DOWN", method="median",
                        buffer_type="frames", tolerance=0.9, threadcount=50):
    
    """
    Creates a stacked flat frame for each chop state. They are either
    combined into a single flat or returned separately.
    
    Parameters:
    ----------------------
    files: list or array 
        List of raw file paths to stack.
    chops: list or array
        List of chop states corresponding to the file list, entries are
        either "CHOP_A" or "CHOP_B".
    chop_direction (optional): string
        The chopping direction employed, either "UP-DOWN", "LEFT-RIGHT",
        or "SEPARATE" if separate flats for each chop are desired.
        Default value is "UP-DOWN", used for single-sided imaging.
    method (optional): string
        Method to integrate images, either by taking the "mean"
        or "median". Default is "median".
    buffer_type (optional): string
        Specifies which type of buffer to use, either "files" or
        "frames". Set to "frames" by default. Note that using median
        integration and file buffers are incompatible.
    tolerance (optional): float
        Fraction of available memory to be used for integration.
        Default value is 0.9.
    threadcount (optional): integer
        Number of threads to employ in multithreading.
        Default value is 50 threads.
    
    Returns: 
    ---------------------- 
    flat: 2D image array
        Stacked flat frame.
    chopa_integrated: 2D image array
        Stacked flat frame for CHOP_A
    chopb_integrated: 2D image array
        Stacked flat frame for CHOP_B
    """

    if buffer_type == "frames":
        chopa_integrated = hf.integrate_frames_buffer(files[chops=="CHOP_A"], method=method,
                                                      tolerance=tolerance, threadcount=threadcount)
        chopb_integrated = hf.integrate_frames_buffer(files[chops=="CHOP_B"], method=method,
                                                      tolerance=tolerance, threadcount=threadcount)
    else:        
        if method == "mean":
            chopa_integrated = hf.integrate_files_buffer(files[chops=="CHOP_A"], tolerance=tolerance,
                                                         threadcount=threadcount)
            chopb_integrated = hf.integrate_files_buffer(files[chops=="CHOP_B"], tolerance=tolerance,
                                                         threadcount=threadcount)
        else:
            raise ValueError("Incompatible integration method and buffer type")

    # Open a file and check file_size and image shape
    hdul = fits.open(files[0])
    array_shape = np.shape(hdul[0].data)
    hdul.close()

    # Force array shape to have correct dimensions
    if len(array_shape) == 3:
        array_shape = (array_shape[1], array_shape[2])
    
    # Stitch chops together such that the halves of the image without the target are combined
    if chop_direction == "UP-DOWN":

        flat = np.concatenate((chopb_integrated[:int(array_shape[0]/2), :],
                               chopa_integrated[int(array_shape[0]/2):, :]))

    elif chop_direction == "LEFT-RIGHT":

        flat = np.concatenate((chopb_integrated[:, :int(array_shape[1]/2)].T,
                               chopa_integrated[:, int(array_shape[1]/2):].T)).T

    else:

        raise ValueError("Invalid chop direction")

    # Remove zero values from the flat, replace with the minimum value    
    flat[flat == 0] = np.min(flat[flat != 0]) 

    # Remove zero values from the flat, replace with the minimum value    
    chopa_integrated[chopa_integrated == 0] = np.min(chopa_integrated[chopa_integrated != 0]) 
    chopb_integrated[chopb_integrated == 0] = np.min(chopb_integrated[chopb_integrated != 0]) 

    return flat, chopa_integrated, chopb_integrated

def create_star_mask(chopa_star_img, chopb_star_img, chopa_flat, chopb_flat,
                     badmap, highpassmask, smooth=2, sigma=7, growth=3):
    """
    Creates a star mask from images containing stars and flats.
    
    Parameters:
    ----------------------
    chopa_flat: 2D image array
        Stacked flat frame for CHOP_A
    chopb_flat: 2D image array
        Stacked flat frame for CHOP_B
    badmap: 2D image array
        Bad pixel map, where bad pixels are set to 0 and all other
        pixels are set to 1.
    highpassmask: 2D numpy array
        mask applied to images before high pass filtering,
        affected pixels are set to np.nan
    smooth (optional): integer
        Radius of smoothing kernel, divided by 5
    sigma (optional): float
        Cutoff value for isolating the star, in standard deviations.
    growth (optional): float
        Star mask growth factor.
        
    Returns: 
    ---------------------- 
    chopa_starpos: 2D image array
        CHOP A star mask.
    chopb_starpos: 2D image array
        CHOP B star mask.
    """
    
    # Create star image with flats
    chopa_starpos = chopa_star_img/chopa_flat - chopb_star_img/chopb_flat

    # Set problematic pixels to nans
    chopa_starpos[(badmap == 0) | (highpassmask == 0)] = np.nan
    
    chopb_starpos = -1*np.copy(chopa_starpos)
    
    chopa_starpos = median_filter(chopa_starpos, size=smooth*6)
    chopb_starpos = median_filter(chopb_starpos, size=smooth*6)

    # Highpass filter and then smooth out any remaining spurious pixels
    chopa_bg_model = convolve_fft(np.pad(chopa_starpos, 10*smooth, mode='edge'),
                                  Ring2DKernel(5*smooth, 4*smooth))[10*smooth:-10*smooth,
                                                                    10*smooth:-10*smooth]

    chopb_bg_model = convolve_fft(np.pad(chopb_starpos, 10*smooth, mode='edge'),
                              Ring2DKernel(5*smooth, 4*smooth))[10*smooth:-10*smooth,
                                                                10*smooth:-10*smooth]

    chopa_starpos = convolve_fft(chopa_starpos-chopa_bg_model, Box2DKernel(smooth*3))
    chopb_starpos = convolve_fft(chopb_starpos-chopb_bg_model, Box2DKernel(smooth*3))

    # Create the mask
    chopa_starpos[chopa_starpos > np.nanmedian(chopa_starpos) + sigma*np.nanstd(chopa_starpos)] = 1
    chopb_starpos[chopb_starpos > np.nanmedian(chopb_starpos) + sigma*np.nanstd(chopb_starpos)] = 1

    chopa_starpos[chopa_starpos < 1] = 0
    chopb_starpos[chopb_starpos < 1] = 0
    
    # Grow the mask
    chopa_starpos = convolve_fft(chopa_starpos, Gaussian2DKernel(growth))
    chopb_starpos = convolve_fft(chopb_starpos, Gaussian2DKernel(growth))

    chopa_starpos[chopa_starpos > 0.1] = 1
    chopb_starpos[chopb_starpos > 0.1] = 1

    chopa_starpos[chopa_starpos < 1] = 0
    chopb_starpos[chopb_starpos < 1] = 0

    return chopa_starpos, chopb_starpos

def subtract_psfs(files, chops, stellar_temp,
                  maxima=None, badmap=None, starmask=None, flats=None, windowsize=35,
                  nbg=1, smooth=5, recur_iteration=2, remove_trefoil=True, remove_residual=False,
                  fit_reject_criterion=100, prefix='', threadcount=50):

    """
    Subtracts the stellar PSF from every image.
    
    Parameters (contained inside a tuple):
    ----------------------
    files: list or array 
        List of raw file paths, sorted 
    chops: string array
        List of chop states corresponding to the file list,
        entries are either "CHOP_A" or "CHOP_B"
    stellar_temp: float
        Temperature of the star in Kelvins.
    maxima (optional): float tuple array
        Tuples encoding location of the PSF in the images
    badmap (optional): 2D image array
        Bad pixel map, where bad pixels are set to 0 and all other
        pixels are set to 1.
    starmask (optional): List of 2D boolean array
        Star masks applied to images for detecting the star,
        affected pixels are set to np.nan
    flats (optional): List of 2D numpy arrays
        Temporary flats applied to locate the star
    windowsize (optional): integer
        Half width/height of the cutout image
        (which is 1:1 aspect ratio)
    nbg (optional): integer
        Number of frames to use in rolling background subtraction
    smooth (optional): integer
        Radius of smoothing kernel, divided by 5
    recur_iteration (optional): integer
        Number of recursive iterations allowed in psf fitting,
        default is 2.
    remove_trefoil (optional): boolean
        Enables removal of psf residual from trefoil.
        Enabled by default.
    remove_residual (optional): boolean
        Enables removal of psf residual through highpass filtering.
        Disabled by default.
    fit_reject_criterion (optional): integer
        Maximum allowed failcode, used in replacing failed psf fits.
    prefix (optional): string
        Prefix to add to directory name when saving image.
    threadcount (optional): integer
        Number of threads to employ in multithreading.
        Default value is 50 threads.

    Returns: 
    ---------------------- 
    psf_subtracted_files: 1D numpy array
        List of psf subtracted file paths, sorted
    maxima: float tuple array
        Tuples encoding location of the PSF in the images, with
        improved accuracy
    failcodes: 1D integer numpy array
        List of integers encoding the success of each PSF fit.
        111: Fitting has failed
        011: Only basic airy fitting has succeeded
        010: Only empirical psf fitting has failed
        001: Only trefoil fitting has failed
        000: Fitting succeeded 
    """

    # Create directory to save files
    root_dir = os.path.dirname(os.path.dirname(files[0]))
    psf_subtracted_dir = os.path.join(root_dir, prefix+'psfsubtracted')
    if not os.path.exists(psf_subtracted_dir):
        os.makedirs(psf_subtracted_dir)

    wvl_interp, relative_flux = hf.calculate_expected_flux(stellar_temp)

    #if __name__ == "__main__":
    with Pool(threadcount) as pool:

        (max_x,
         max_y,
         failcodes,
         reffits,
         lbtfits,
         trifits) = zip(*tqdm(pool.imap(PSFSubtraction((psf_subtracted_dir, files,
                                                        chops, maxima, badmap, starmask,
                                                        flats, wvl_interp, relative_flux,
                                                        windowsize, nbg, smooth, recur_iteration,
                                                        remove_trefoil, remove_residual)),
                                        range(len(files))), total=len(files),
                              desc="Subtracting PSFs"))

    failcodes, reffits, lbtfits, trifits = (np.asarray(failcodes), np.asarray(reffits),
                                            np.asarray(lbtfits), np.asarray(trifits))
    
    maxima = np.vstack((np.asarray(max_x), np.asarray(max_y))).T

    failed_indices = np.arange(len(files))[failcodes > fit_reject_criterion]

    if len(failed_indices) > 0:
        #if __name__ == "__main__":
        with Pool(threadcount) as pool:
    
            (new_max_x,
             new_max_y) = zip(*tqdm(pool.imap(PSFSubRedux((psf_subtracted_dir, files,
                                                           chops, maxima, failcodes,
                                                           reffits, lbtfits, trifits, badmap,
                                                           flats, wvl_interp, relative_flux,
                                                           windowsize, fit_reject_criterion,
                                                           nbg, smooth, remove_residual)),
                                              failed_indices), total=len(failed_indices),
                                    desc="Subtracting failed PSFs"))
            
        maxima[failed_indices] = np.vstack((np.asarray(new_max_x), np.asarray(new_max_y))).T

    # Important to sort the data for reading it sequentially
    psf_subtracted_files = sorted(list(pathlib.Path(str(psf_subtracted_dir)).rglob('*.fits')))
    psf_subtracted_files = np.asarray([a for a in psf_subtracted_files\
                                       if a.name[0]!='.'\
                                       and str(a.parent)==psf_subtracted_dir])

    return psf_subtracted_files, maxima, failcodes, reffits, lbtfits, trifits

def parallelized_chop_subtraction(files, chops, prefix='', threadcount=50):

    """
    Parallelization of pyNOMIC.helper_functions.chop_subtraction
    for batched chop subtraction.

    Parameters:
    ----------------------
    files: list or array 
        List of raw file paths, sorted 
    chops: string array
        List of chop states corresponding to the file list,
        entries are either "CHOP_A" or "CHOP_B"
    prefix (optional): string
        Prefix to add to directory name when saving image.
    threadcount (optional): integer
        Number of threads to employ in multithreading.
        Default value is 50 threads.

    Returns:
    ----------------------    
    chop_subtracted_files: 1D numpy array
        List of chop subtracted file paths, sorted
    """
    
    # Create directory to save files
    root_dir = os.path.dirname(os.path.dirname(files[0]))
    chop_subtracted_dir = os.path.join(root_dir, prefix+'chopsubtracted')

    if not os.path.exists(chop_subtracted_dir):
        os.makedirs(chop_subtracted_dir)

    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        offset_x, offset_y = zip(*tqdm(pool.imap(ChopSubtract((files, chops, chop_subtracted_dir)),
                                                 range(len(files))), total=len(files),
                                       desc = "Chop subtracting frames..."))
    # Get file paths
    chop_subtracted_files = sorted(list(pathlib.Path(str(chop_subtracted_dir)).rglob('*.fits')))
    chop_subtracted_files = np.asarray([a for a in chop_subtracted_files if a.name[0] != '.'\
                                        and str(a.parent) == chop_subtracted_dir])
    
    return chop_subtracted_files

def chop_align(files, chops, interp_method="linear", ref_index=None, smooth=3,
               channel_edges=[127, 255, 383], prefix='', threadcount=50):
    
    """
    Aligns background of PSF subtracted files using FFT alignment.

    Parameters:
    ----------------------
    files: list or array 
        List of file paths, sorted 
    chops: string array
        List of chop states corresponding to the file list,
        entries are either "CHOP_A" or "CHOP_B"
    interp_method: string
        Interpolation method for
        scipy.interp.RegularGridInterpolator,
        default is cubic interpolation. Linear interpolation is much
        faster but imprecise especially at the center of the PSF.
    ref_index: integer
        File index of the reference. If None, the index used is the
        median of the sequence.
    smooth: integer
        Radius of smoothing kernel. Default is 3 pixels.
    channel_edges: list of integers
        List containing indices corresponding to the central row of
        channel edges
    prefix (optional): string
        Prefix to add to directory name when saving image.
    threadcount (optional): integer
        Number of threads to employ in multithreading.
        Default value is 50 threads.

    Returns:
    ----------------------    
    chopres_files: 1D numpy array
        List of chop residual aligned file paths, sorted
    highfreq_files: 1D numpy array
        List of file paths with low frequency information, including
        chop residuals removed, sorted
    chopres_stds: 1D numpy array
        Standard deviations of the chop residual maps.
    highfreq_stds: 1D numpy arrays
        Standard deviations of the high frequency components.
    offsets: 2D numpy array
        List of calculated offsets from FFT alignment.
    """
    
    # Create directory to save files
    root_dir = os.path.dirname(os.path.dirname(files[0]))
    chopres_dir = os.path.join(root_dir, prefix+'chopres_align')
    highfreq_dir = os.path.join(root_dir, prefix+'highfreq')
    
    if not os.path.exists(chopres_dir):
        os.makedirs(chopres_dir)
    if not os.path.exists(highfreq_dir):
        os.makedirs(highfreq_dir)

    # Default to middle index
    if ref_index is None:
        ref_index = int(len(files)/2)

    hdul = fits.open(files[ref_index])
    img = hdul[0].data
    hdul.close()

    # Create alignment grid
    frameh, framew = np.shape(img)
    px = np.linspace(0, framew-1, framew)
    py = np.linspace(0, frameh-1, frameh)

    # Chop subtract reference
    reference = hf.chop_subtraction(img, ref_index, chops[ref_index], files, [0,0], 1,
                                    correction_method="subtraction")
    # Repair channel edges
    for channel_edge in channel_edges:
        reference = hf.repair_channel_edges(reference, channel_edge)
    
    # Convolve frame
    reference = convolve_fft(np.pad(reference, 10*smooth, mode='edge'),
                             Gaussian2DKernel(smooth))[10*smooth:-10*smooth,
                                                       10*smooth:-10*smooth]
    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        (offset_x,
         offset_y,
         chopres_stds,
         highfreq_stds) =zip(*tqdm(pool.imap(ChopAlign((files, chops, chopres_dir, highfreq_dir,
                                                            reference, px, py, ref_index, smooth,
                                                            channel_edges, interp_method)),
                                      range(len(files))), total=len(files),
                            desc = "Aligning frames..."))

    offsets = np.vstack((np.asarray(offset_x), np.asarray(offset_y))).T
    
    # Get file paths
    chopres_files = sorted(list(pathlib.Path(str(chopres_dir)).rglob('*.fits')))
    chopres_files = np.asarray([a for a in chopres_files if a.name[0]!='.'\
                                and str(a.parent)==chopres_dir])

    # Get file paths
    highfreq_files = sorted(list(pathlib.Path(str(highfreq_dir)).rglob('*.fits')))
    highfreq_files = np.asarray([a for a in highfreq_files if a.name[0]!='.'\
                                and str(a.parent)==highfreq_dir])
    
    return chopres_files, highfreq_files, chopres_stds, highfreq_stds, offsets
        
def subtract_background(raw_files, psf_subtracted_files, chops, channel_edges=[127, 255, 383],
                        biased_columns=[303], biased_rows = [], striped_regions=[],
                        vertical_biases=[], horizontal_biases=[], nanrows=[], nancols=[],
                        flats=None, resflats=None, flat_offsets=None,
                        correction_method="division", channel_method="linear",
                        nbg=1, smooth=5, edge_cut = 2, prefix="", threadcount=50):
    
    """
    Subtract background from adjacent chop frames, divide by flat,
    perform highpass filter

    Parameters:
    ----------------------
    raw_files: list or array 
        List of raw file paths, sorted 
    psf_subtracted_files: list or array 
        List of psf subtracted file paths, sorted 
    chops: string array
        List of chop states corresponding to the file list,
        entries are either "CHOP_A" or "CHOP_B"
    channel_edges (optional): list
        List containing indices corresponding to channel edges
    biased_columns (optional): list
        List containing indices corresponding to biased columns
    striped_regions (optional): list
        List containing indices corresponding to regions of
        the image needing destriping. Each entry contains four integers
        for slicing the image: [0:1, 2:3]
    vertical_biases (optional): list
        List containing indices corresponding to columns separating
        regions of the image with different biases
    horizontal_biases (optional): list
        List containing indices corresponding to rows separating
        regions of the image with different biases
    biased_rows (optional): list
        List containing indices corresponding to biased rows
    nanrows (optional): list
        List containing indices corresponding to rows that
        need to be set to np.nan
    nancols (optional): list
        List containing indices corresponding to columns that
        need to be set to np.nan
    flats (optional): list
        List containing the flats for each chop state
    resflats (optional): list
        List containing low frequency flats for correcting chop 
        residuals for each chop state.
    flat_offsets (optional): 2D numpy array
        List of tuples containing the offsets of the background
        with respect to the flat.
    correction_method (optional): string
        Method to subtract background model/flat, default is "division".
        Other options include "subtraction".
    channel_method (optional): string
        Method to correct channel edges,
        either "linear" or "nearestneighbor" interpolation.
    nbg (optional): integer
        Number of frames to use in rolling background subtraction
    smooth (optional): integer
        Radius of smoothing kernel, divided by 5
    edge_cut (optional): integer
        Number of pixels to remove at the edges of images
    prefix (optional): string
        Prefix to add to directory name when saving image.
    threadcount (optional): integer
        Number of threads to employ in multithreading.
        Default value is 50 threads.

    Returns:
    ----------------------    
    subtracted_dir: string/Path object
        Directory where subtracted images are saved
    """
    
    # Create directory to save files
    root_dir = os.path.dirname(os.path.dirname(raw_files[0]))
    subtracted_dir=os.path.join(root_dir, prefix+'subtracted')
    
    if not os.path.exists(subtracted_dir):
        os.makedirs(subtracted_dir)
        
    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        (chopsub_stds,
         backsub_stds) = zip(*tqdm(pool.imap(SubtractBackground((subtracted_dir, raw_files,
                                                                 psf_subtracted_files, chops,
                                                                 channel_edges, biased_columns,
                                                                 striped_regions, vertical_biases,
                                                                 horizontal_biases, biased_rows, nanrows,
                                                                 nancols, flats, resflats,
                                                                 flat_offsets, correction_method,
                                                                 channel_method, nbg, smooth, edge_cut)),
                                             range(len(raw_files))), total=len(raw_files),
                                   desc = "Subtracting backgrounds..."))

    return np.asarray(chopsub_stds), np.asarray(backsub_stds)

def frame_registration(files, subtracted_dir, maxima=None, badmap=None, starmask=None,
                       alignment_method="fitting", interp_method="cubic", windowsize=20,
                       stellar_temp=5778, actmax=False, model_trefoil=True, boxcar=False,
                       save_files=True, prefix='', threadcount=50):

    """
    Aligns all frames together, centering the PSF in the middle
    of the image by translation and padding.
    
    Parameters:
    ----------------------
    files: list or array 
        List of file paths, sorted 
    subtracted_dir: string/Path object
        Directory where images are read from
    maxima (optional): float tuple array
        Tuples encoding location of the PSF in the images
    badmap (optional): 2D image array
        Bad pixel map, where bad pixels are set to 0 and all other
        pixels are set to 1.
    starmask (optional): 2D boolean array
        mask applied to image to allow for better star detection,
        affected pixels are set to np.nan
    alignment_method (optional): string
        Method by which to align PSFS, either through FFT transform
        ("fft"), or through airy disk fitting ("fitting"). Default
        is FFT.
    interp_method (optional): string
        Interpolation method for scipy.interp.RegularGridInterpolator,
        default is cubic interpolation. Linear interpolation is much
        faster but imprecise especially at the center of the PSF.
    windowsize (optional): integer
        Half width/height of the reference cutout image
        (which is 1:1 aspect ratio). Default is 20 pixels.
    stellar_temp (optional): float
        Temperature of the star in Kelvins.
    actmax (optional): boolean
        Skips subpixel PSF measurement and instead accepts the positions
        provided by the variable "maxima" as the true location. Disabled
        by default.
    model_trefoil (optional): boolean
        Enables the modeling of trefoil in the PSF. Enabled by default.
    boxcar (optional): boolean
        Enables 2x2 boxcar smoothing to combat odd/even detector
        noise. False by default.
    save_files: boolean
        Enables saving the aligned files, True by default.
    prefix (optional): string
        Prefix to add to directory name when saving image.
    threadcount (optional): integer
        Number of threads to employ in multithreading.
        Default value is 50 threads.
        
    Returns:
    ----------------------    
    original_psf_locs: 2 x len(files) numpy array
        Array containing pixel coordinates of psf locations
        in the input images    
    imgfits: List of 1D numpy arrays or 2D numpy array
        If airy disk fitting:
        Array containing airy disk fit parameters for each file 
        If FFT fitting:
        List containing fit parameters for the reference image
    array_shape: integer tuple
        Tuple containing image dimensions, from numpy.shape
    file_size: float
        File size of aligned images.
    aligned_files: list or array
        List of aligned file paths, sorted 
    """
            
    print("Creating alignment grid ...")    

    # Create directory to save files
    root_dir = os.path.dirname(subtracted_dir)
    aligned_dir = os.path.join(root_dir, prefix+'aligned')
    if not os.path.exists(aligned_dir):
        os.makedirs(aligned_dir)

    # Open two frames (of different chop states)
    hdul = fits.open(os.path.join(subtracted_dir, "subtracted_"+files[0].name))
    hdul2 = fits.open(os.path.join(subtracted_dir, "subtracted_"+files[1].name))
    
    first_frame = hdul[0].data
    sec_frame = hdul2[0].data
    array_shape = np.shape(first_frame)

    if maxima is None:
        # Find maxima and minima of both frames
        test_first_frame = np.copy(first_frame)
        test_sec_frame = np.copy(sec_frame)
        if badmap is not None:
            test_first_frame[(badmap == 0)] = np.nan
            test_sec_frame[(badmap == 0)] = np.nan
        if starmask is not None:
            test_first_frame[(starmask == 0)] = np.nan
            test_sec_frame[(starmask == 0)] = np.nan
        first_maximum = np.asarray(np.where(test_first_frame == np.nanmax(test_first_frame)))[:,0]
        sec_maximum = np.asarray(np.where(test_sec_frame == np.nanmax(test_sec_frame)))[:,0]
    else:
        first_maximum = maxima[0]
        sec_maximum = maxima[1]

    first_bound = (int(first_maximum[0])+windowsize) - array_shape[0]
    sec_bound = (int(sec_maximum[0])+windowsize) - array_shape[0]
    if (first_bound < 0) & (first_bound > (2*windowsize - array_shape[0])):
        first_bound = 0
    elif (first_bound <= (2*windowsize  - array_shape[0])):
        first_bound = int(maximum[0]) - windowsize
    if (sec_bound < 0) & (sec_bound > (2*windowsize - array_shape[0])):
        sec_bound = 0
    elif (sec_bound <= (2*windowsize  - array_shape[0])):
        sec_bound = int(maximum[0]) - windowsize
    first_maximum[0] -= first_bound
    sec_maximum[0] -= sec_bound

    hdul.close()
    hdul2.close()

    # Get shape of frame
    frameh, framew = np.shape(first_frame)

    # Create x, y lists for the image
    x = np.linspace(0, framew-1, framew)
    y = np.linspace(0, frameh-1, frameh)
        
    # Create meshgrid window for subpixel alignment of PSF
    wx = np.linspace(0, 2*windowsize-1, 2*windowsize)
    wy = np.linspace(0, 2*windowsize-1, 2*windowsize)
    wx, wy = np.meshgrid(wx, wy)

    # Get cutouts of PSFs in both frames
    first_cutout = first_frame[(int(first_maximum[0])-windowsize):(int(first_maximum[0])+windowsize),
                               (int(first_maximum[1])-windowsize):(int(first_maximum[1])+windowsize)]
    sec_cutout = sec_frame[(int(sec_maximum[0])-windowsize):(int(sec_maximum[0])+windowsize),
                           (int(sec_maximum[1])-windowsize):(int(sec_maximum[1])+windowsize)]

    if actmax is False:
            
        wvl_interp, relative_flux = hf.calculate_expected_flux(stellar_temp)
        
        # Fit an airy disk to the PSF of the first frame
        reffit, lbtfit, trifit = hf.empirical_psf_fit(first_cutout, wvl_interp, relative_flux,
                                                      model_trefoil=model_trefoil)
        '''
        Create airy disk reference for aligning frames,
        but centered exactly in the middle of the cutout
        '''
        reference = lbtfit[0]*np.mean(hf.modified_airy_disk((wx, wy), relative_flux, wvl_interp,
                                                            lbtfit[2], lbtfit[3], 0, reffit[4],
                                                            windowsize-0.5, windowsize-0.5),
                                      axis=0) + lbtfit[1]
    
        if model_trefoil and ~np.isnan(trifit[0]):
            reference +=  hf.center_triangle((wx, wy), trifit[0], trifit[1], trifit[2],
                                             trifit[3], trifit[4], windowsize-0.5,
                                             windowsize-0.5, trifit[5], trifit[6], ravel=False)
                
        if alignment_method == "fft":
        
            # Get the offsets of the cutouts from the reference 
            first_offset = chi2_shift(reference, first_cutout, upsample_factor='auto',
                                      return_error=False)
            sec_offset = chi2_shift(reference, sec_cutout, upsample_factor='auto',
                                       return_error=False)
            first_offset = np.asarray([first_offset[0], first_offset[1]])
            sec_offset = np.asarray([sec_offset[0], sec_offset[1]])
    
        elif alignment_method == "fitting":
    
            # Run curve_fit to get airy best fit parameters
            reffit, _ = curve_fit(hf.airy_disk, (wx, wy), first_cutout.ravel(),
                                  p0=[np.max(first_cutout), 30, 30, -1, 0, windowsize-0.5,
                                      windowsize-0.5+first_bound],
                                  bounds=([0, 1, 1, 1*-np.inf, 0, 1, 1],
                                          [10*np.max(first_cutout), 200, 200, np.inf, 2*np.pi,
                                           2*windowsize, 2*windowsize]))
    
            # Run curve_fit to get airy best fit parameters
            sec_reffit, _ = curve_fit(hf.airy_disk, (wx, wy), sec_cutout.ravel(),
                                      p0=[np.max(sec_cutout), 30, 30, -1, 0, windowsize-0.5,
                                          windowsize-0.5+sec_bound],
                                      bounds=([0, 1, 1, 1*-np.inf, 0, 1, 1],
                                              [10*np.max(sec_cutout), 200, 200, np.inf, 2*np.pi,
                                               2*windowsize, 2*windowsize]))
    
            first_offset = np.asarray([reffit[5] - (windowsize - 0.5),
                                       reffit[6] - (windowsize - 0.5)])
            sec_offset = np.asarray([sec_reffit[5] - (windowsize - 0.5),
                                        sec_reffit[6] - (windowsize - 0.5)])
        else:
            raise ValueError("Invalid alignment method")
    else:
        first_offset = np.asarray([maxima[0][1] - int(maxima[0][1]),
                                   maxima[0][0] - int(maxima[0][0])])
        sec_offset = np.asarray([maxima[1][1] - int(maxima[1][1]),
                                 maxima[1][0] - int(maxima[1][0])])
        
    # Include offset of second image from first image
    sec_offset[0] += (int(sec_maximum[1]) - int(first_maximum[1]))
    sec_offset[1] += (int(sec_maximum[0]) - int(first_maximum[0]))

    # Calculate necessary padding to align second image
    padding = np.asarray([int(np.ceil(np.abs(sec_offset[0]))*np.sign(sec_offset[0])),
                          int(np.ceil(np.abs(sec_offset[1]))*np.sign(sec_offset[1]))])

    # Create new x, y lists including padding
    px = np.linspace(0, framew-1+np.abs(padding[0]), framew+np.abs(padding[0]))
    py = np.linspace(0, frameh-1+np.abs(padding[1]), frameh+np.abs(padding[1]))

    # Align first frame
    first_frame = hf.align_frame(first_frame, px, py, -1*padding, first_offset - padding,
                                 method=interp_method)

    # Calculate the origin of the frame after alignment
    origin = np.asarray([int(first_maximum[1])-0.5, int(first_maximum[0])-0.5])

    # Include the padding required for the second frame in the calculation of the origin
    if padding[1] >= 0:
        origin[1] = origin[1] + padding[1]
    if padding[0] >= 0:
        origin[0] = origin[0] + padding[0]

    # Calculate padding required to center the PSF in the frame
    center_padding = (-1*int(2*origin[0] - np.shape(first_frame)[1] + 1),
                      -1*int(2*origin[1] - np.shape(first_frame)[0] + 1))

    if save_files:
        
        # Pad first frame
        first_frame = hf.pad_frame(first_frame, len(px) + np.abs(center_padding[0]),
                                   len(py) + np.abs(center_padding[1]), center_padding)
    
        # Align second frame
        sec_frame = hf.align_frame(sec_frame, px, py, padding, sec_offset,
                                      method=interp_method)
        sec_frame = hf.pad_frame(sec_frame, len(px) + np.abs(center_padding[0]),
                                    len(py) + np.abs(center_padding[1]), center_padding)
    
        if boxcar:
            first_frame = convolve_fft(first_frame, Box2DKernel(2), preserve_nan=True)
            sec_frame = convolve_fft(sec_frame, Box2DKernel(2), preserve_nan=True)
        
        # Save first frame
        newhdul = fits.HDUList([fits.PrimaryHDU(data=(first_frame))])
        newhdul.writeto(os.path.join(aligned_dir, "aligned_"+files[0].name), overwrite=True)
    
        # Save second frame
        newhdul = fits.HDUList([fits.PrimaryHDU(data=(sec_frame))])   
        newhdul.writeto(os.path.join(aligned_dir, "aligned_"+files[1].name), overwrite=True)
        newhdul.close()

    # Align the rest of the images
    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        offsets, imgfits = zip(*tqdm(pool.imap(RegisterFrames((files, subtracted_dir, aligned_dir,
                                                               badmap, starmask, padding,
                                                               center_padding, px, py, wx, wy,
                                                               reference, first_maximum, maxima,
                                                               windowsize, alignment_method,
                                                               interp_method, actmax, 
                                                               boxcar, save_files)),
                                               range(len(files))[2:]), total=len(files) - 2,
                                     desc="Aligning frames"))

    # Include first two frame offsets
    offsets = np.concatenate((np.asarray([(first_offset[0], first_offset[1]),
                                          (sec_offset[0], sec_offset[1])]),
                              np.asarray(offsets)))
    if alignment_method == "fitting":
        imgfits = np.concatenate((np.asarray([reffit, sec_reffit]),
                              np.asarray(imgfits)))
    else:
        imgfits = (reffit, lbtfit, trifit)

    pad_array = np.asarray([padding[0]*np.power(-1, np.arange(np.shape(offsets)[0])),
             padding[1]*np.power(-1, np.arange(np.shape(offsets)[0]))])
    pad_array[:, 0] *= -1

    # Calculate the location of the PSF in the original unaligned images
    original_psf_locs = np.asarray([((framew + pad_array[1] - center_padding[1])/2 +
                                     offsets[:,1] - 0.5),
                                    ((frameh + pad_array[0] - center_padding[0])/2 +
                                     offsets[:,0] - 0.5)]).T

    if save_files:
        # Get file paths
        aligned_files = sorted(list(pathlib.Path(str(aligned_dir)).rglob('*.fits')))
        aligned_files = np.asarray([a for a in aligned_files if a.name[0]!='.'\
                                    and str(a.parent)==aligned_dir])
    else:
        aligned_files = None
        
    return (original_psf_locs, imgfits, np.shape(first_frame),
            float(first_frame.nbytes), aligned_files)

def mask_files(files, directory, locs, array_shape, nbg=1, nan_mask_radius = 16, threadcount=50):
    
    """
    Mask out circular regions of the image at certain locations.

    Parameters:
    ----------------------
    files: list or array 
        List of file paths, sorted 
    directory: string/Path object
        Directory where images will be saved
    locs: 2 x len(files) numpy array
        Array containing pixel PSF coordinates   
    array_shape: integer tuple
        Tuple containing image dimensions, from numpy.shape
    nbg (optional): integer
        Number of adjacent frames to use to locate the 
        oversubtracted psf
    nan_mask_radius (optional): integer
        Radius of the nan mask
    threadcount (optional): integer
        Number of threads to employ in multithreading.
        Default value is 50 threads.
        
    Returns:
    ----------------------    
    masked_files: list or array
        List of file paths to the masked files, sorted
    """

    if not os.path.exists(directory):
        os.makedirs(directory)

    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        results = zip(*tqdm(pool.imap(hf.MaskFrames((files, directory, locs, array_shape, nbg,
                                                     nan_mask_radius)),
                                      range(len(files))), total=len(files),
                           desc="Masking files"))

    masked_files = sorted(list(pathlib.Path(directory).rglob('*.fits')))
    masked_files = np.asarray([a for a in masked_files if a.name[0]!='.'\
                               and str(a.parent)==directory])
    
    return masked_files

