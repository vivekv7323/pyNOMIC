#----------------------------------------
# IMPORTS
#----------------------------------------
import os
import numpy as np
from tqdm.auto import tqdm

from astropy.io import fits
from astropy.table import QTable
import astropy.constants as c
import astropy.units as u
from astropy.convolution import convolve_fft, Gaussian1DKernel

from scipy.special import j1
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator

#----------------------------------------
# CLASSES
#----------------------------------------

class IntegrateFrames(object):
    
    """
    Simple class to parallelize image integration with nans.
    """

    def __init__(self, params):
        
        """
        Parameters (contained inside a tuple):
        ----------------------
        array_shape: integer tuple
            Tuple containing image dimensions, from numpy.shape
        """
        
        self.params = params
    
    def __call__(self, files):

        """
        Parameters:
        ----------------------
        files: list or array 
            List of raw file paths to stack

        Returns: 
        ---------------------- 
        stacked_img: 2D image array
            Integrated stack of frames (sum)
        count: integer
            Number of frames
        """

        array_shape = self.params
        
        # array for integrating files      
        buf_3D = np.zeros((len(files), array_shape[0], array_shape[1]))
        
        count = 0
        
        for k in range(len(files)):

            hdul = fits.open(files[k])
        
            buf_3D[k] = hdul[0].data
    
            hdul.close()
    
            count += 1
            
        stacked_img = np.nanmean(buf_3D, axis=0)*count
        
        return stacked_img, count

class MaskFrames(object):

    '''
    Class to apply a circular nan mask onto the oversubtracted PSF
    in background subtracted images. 
    '''
    
    def __init__(self, params):

        """
        Parameters (contained inside a tuple):
        ----------------------
        file: string/Path object
            Path to raw image file
        directory: string/Path object
            Directory where images will be saved
        locs: 2 x len(files) numpy array
            Array containing pixel PSF coordinates   
        array_shape: integer tuple
            Tuple containing image dimensions, from numpy.shape
        nbg: integer
            Number of adjacent frames to use to locate the 
            oversubtracted psf
        nan_mask_radius: integer
            Radius of the nan mask
        """
        
        self.params = params
    
    def __call__(self, i):

        """
        Parameters:
        ----------------------
        index: integer
            File index to process from 'files'        
        """

        files, directory, locs, array_shape, nbg, nan_mask_radius = self.params

        hdul = fits.open(files[i])

        frame = hdul[0].data

        origin = [array_shape[0]/2 - 0.5, array_shape[1]/2 - 0.5]

        # Locate adjacent images of the same chop state
        bg_indices = np.arange(i - 2*nbg + 1, i + 2*nbg, 2)
        
        # Exclude nonexistent indices and the current index
        bg_indices = bg_indices[(bg_indices >= 0) & (bg_indices < len(files)) & (bg_indices != i)]

        translation = (np.mean(locs[bg_indices], axis=0) - locs[i])
        
        frame[circular_mask((origin[0] + translation[0], origin[1] + translation[1]),
                            nan_mask_radius, array_shape[1], array_shape[0])] = np.nan

        # Write image to path
        newhdul = fits.HDUList([fits.PrimaryHDU(data=frame)])
        newhdul.writeto(os.path.join(directory, "masked_"+str(files[i].name)), overwrite=True)
        newhdul.close()

        return True, True

            
#----------------------------------------
# FUNCTIONS
#----------------------------------------

def spatial_binning(img_cube, spatial_bin):

    """
    Bins images spatially in an image cube.
    
    Parameters:
    ----------------------
    img_cube: 3D numpy array
        A numpy array of dimensions N x W X H, containing N images
    spatial_bin: integer
        Integer binning factor
        
    Returns:
    ----------------------    
    new_cube: 3D numpy array
        Binned output array containing N images    
    array_shape: integer tuple
        Shape of new_cube
    """

    if spatial_bin != 1:
        
        shape = np.shape(img_cube)
    
        # Find desired array shape based on the spatial bin required
        # and the requirement of even image sizes
        array_shape = (shape[0], (2*np.ceil(shape[1]/(2*spatial_bin))).astype(np.int32),
                      (2*np.ceil(shape[2]/(2*spatial_bin))).astype(np.int32)) 
        
        # Pad image cube in preparation for binning to the array shape
        new_cube = np.pad(img_cube, ((0,0),(int(0.5*(spatial_bin*array_shape[1]-shape[1])),
                                            int(0.5*(spatial_bin*array_shape[1]-shape[1]))),
                                           (int(0.5*(spatial_bin*array_shape[2]-shape[2])),
                                            int(0.5*(spatial_bin*array_shape[2]-shape[2])))),
                                            constant_values=np.nan)
        
        #new_cube[new_cube == flag_value] = np.nan
    
        # Perform binning with np.sum and np.reshape
        new_cube = np.sum(np.sum(np.reshape(new_cube, (array_shape[0], array_shape[1],
                          spatial_bin, array_shape[2], spatial_bin)), axis=4), axis=2)

    else:
        
        new_cube = img_cube 
        array_shape = np.shape(img_cube)
        
    return new_cube, array_shape

        
def circular_mask(center, radius, width, height):
    '''
    Creates circular mask of certain radius in an image of
    certain width and height
    '''
    Y, X = np.ogrid[:height, :width]
    distance = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = distance <= radius
    
    return mask

def psf_removal_mask(center, inner_radius, outer_radius, width, height):
    
    """
    Creates a smoothed circular mask.
    
    Parameters:
    ----------------------
    center: integer tuple
        Image location to center the mask.
    inner_radius: float
        Radius at which all pixels within the radius are set to 1.
    outer_radius: float
        Radius at which all pixels outside the radius are set to 0.
    width: int
        Image width.
    height: int
        Image height.
        
    Returns:
    ----------------------    
    mask: 2D numpy array
    """

    # create grid
    Y, X = np.ogrid[:height, :width]

    # Define pixels on distance from center, normalized with inner and outer radius
    distance = (np.sqrt((X - center[0])**2 + (Y-center[1])**2) - inner_radius) / outer_radius

    # Set all values inside outer radius to 0, set all values outside the radius to 1
    distance[distance < 0] = 0
    distance[distance > 1] = 1

    # Reverse image mask
    return 1 - distance


def repairChannelEdges(image, loc, method="linear"):
        
    """
    Fill in data for three horizontal channel edges on the NOMIC
    detector, to prevent artifacting. Each channel edge consists of
    three rows and requires data for each.

    Parameters:
    ----------------------
    image: 2D numpy array
        Input image.
    loc: int
        Location of the central row of the channel edge to be corrected.
    method (optional): string
        Method to correct channel edge. Default is linear interpolation,
        other options include nearest neighbor interpolation
        and gradient modeling.
        
    Returns:
    ----------------------    
    image: 2D numpy array
        Corrected image.
    """

    if method == "gradient":
        
        gauss = Gaussian1DKernel(stddev=5)
        
        # Replicate noise profile of adjacent rows
        image[loc+1,:] *= 0.7*np.std(image[loc+2,:])/np.std(image[loc+1,:])
        image[loc-1,:] *= 0.7*np.std(image[loc-2,:])/np.std(image[loc-1,:])
        
        # For middle row of the channel edge, average the top and bottom
        image[loc,:] *= 0.5*(np.std(image[loc+1,:]) + np.std(image[loc-1,:]))/np.std(image[loc,:])
    
        # Get smoothed difference between channel edge and adjacent row and add it back
        image[loc+1,:] +=  convolve_fft(image[loc+2,:]-image[loc+1,:], gauss)
        image[loc-1,:] +=  convolve_fft(image[loc-2,:]-image[loc-1,:], gauss)
    
        # Get smoothed difference between middle row and averaged top & bottom rows and add it back
        image[loc,:] += convolve_fft( 0.5*(image[loc-1,:] + image[loc+1,:])-image[loc,:], gauss)

    elif method == "nearestneighbor":

        image[loc-1, :] = image[loc-2,:]
        image[loc+1, :] = image[loc+2,:]
        image[loc, :] = image[loc-2,:]*(1/2) + image[loc+2,:]*(1/2)
        
    elif method == "linear":

        image[loc-1, :] = image[loc-2,:]*(3/4) + image[loc+2,:]*(1/4)
        image[loc, :] = image[loc-2,:]*(1/2) + image[loc+2,:]*(1/2)
        image[loc+1, :] = image[loc-2,:]*(1/4) + image[loc+2,:]*(3/4)

    else:
        
        raise ValueError("Invalid method")

    return image


def repairVerticalLine(image, loc, ref=None, return_ref=False, stddev=5):
        
    """
    Repair vertical line artifact in the NOMIC detector while preserving
    the unique noise profile.

    Parameters:
    ----------------------
    image: 2D numpy array
        Input image.
    loc: int
        Location of the column to be corrected.
    ref (optional): 2D numpy array
        Reference image from which the correction is estimated for the 
        input image.
    return_ref (optional): boolean
        Returns the reference with the correction applied.
    stddev: integer
        Radius of the gaussian kernel used to determine the gradient 
        along the column.
        
    Returns:
    ----------------------    
    image: 2D numpy array
        Corrected image.
    """

    if ref is None:
        ref = np.copy(image)
    
    gauss = Gaussian1DKernel(stddev=stddev)
    
    # Average adjacent columns
    true = 0.5*(ref[:, loc+1] + ref[:, loc-1])

    # Get smoothed difference between the column and the averaged adjacent columns and add it back
    diff = convolve_fft(true - ref[:, loc], gauss)

    image[:, loc] += diff

    if return_ref:
        ref[:, loc] += diff
        return image, ref
    else:
        return image


def repairHorizontalLine(image, loc, ref=None, return_ref=False, stddev=5):
        
    """
    Repair horizontal line artifact in the NOMIC detector while
    preserving the unique noise profile.

    Parameters:
    ----------------------
    image: 2D numpy array
        Input image.
    loc: int
        Location of the row to be corrected.
    ref (optional): 2D numpy array
        Reference image from which the correction is estimated for the 
        input image.
    return_ref (optional): boolean
        Returns the reference with the correction applied.
    stddev: integer
        Radius of the gaussian kernel used to determine the gradient 
        along the row.
        
    Returns:
    ----------------------    
    image: 2D numpy array
        Corrected image.
    """
    
    if ref is None:
        ref = np.copy(image)
       
    gauss = Gaussian1DKernel(stddev=stddev)
    
    # Average adjacent columns
    true = 0.5*(ref[loc+1,:] + ref[loc-1,:])

    # Get smoothed difference between the column and the averaged adjacent columns and add it back
    diff = convolve_fft(true - ref[loc,:], gauss)
    
    image[loc,:] += diff
    
    if return_ref:
        ref[loc,:] += diff
        return image, ref
    else:
        return image


def repairVerticalBias(image, loc, ref=None, return_ref=False):
    
    """
    Offset bias between two sides of the detector, column-wise.

    Parameters:
    ----------------------
    image: 2D numpy array
        Input image.
    loc: int
        Location of the column separating the two sides of the detector.
    ref (optional): 2D numpy array
        Reference image from which the correction is estimated for the 
        input image.
    return_ref (optional): boolean
        Returns the reference with the correction applied.
        
    Returns:
    ----------------------    
    image: 2D numpy array
        Corrected image.
    """
    
    if ref is None:
        ref = np.copy(image)
        
    array_shape = np.shape(image)

    # Calculate offsets between the two sides solely based on the 5 adjacent columns on either side
    offsets = np.nanmean(ref[:,int(loc - 6):int(loc - 1)] -
                         ref[:,int(loc):int(loc + 5)], axis=1)

    # Construct a polynomial fit along the rows to model the offset
    xoff = np.linspace(0, len(offsets)-1, len(offsets))
    p = np.polyfit(xoff[~np.isnan(offsets)], offsets[~np.isnan(offsets)], deg=3)

    # Subtract the fit
    diff = np.asarray([p[0]*xoff**3 + p[1]*xoff**2 + p[2]*xoff + p[3]]*int(array_shape[1] - loc)).T

    image[:, int(loc):] += diff

    if return_ref:
        ref[:, int(loc):] += diff
        return image, ref
    else:
        return image
    
def repairHorizontalBias(image, loc, ref=None, return_ref=False):
    
    """
    Offset bias between two sides of the detector, row-wise.

    Parameters:
    ----------------------
    image: 2D numpy array
        Input image.
    loc: int
        Location of the row separating the two sides of the detector.
    ref (optional): 2D numpy array
        Reference image from which the correction is estimated for the 
        input image.
    return_ref (optional): boolean
        Returns the reference with the correction applied.
                
    Returns:
    ----------------------    
    image: 2D numpy array
        Corrected image.
    """
    
    if ref is None:
        ref = np.copy(image)
        
    array_shape = np.shape(image)
    
    # Calculate offsets between the two sides solely based on the 5 adjacent rows on either side
    offsets = np.nanmean(ref[int(loc - 6):int(loc - 1),:] -
                         ref[int(loc):int(loc + 5),:], axis=0)
    
    # Construct a polynomial fit along the columns to model the offset
    xoff = np.linspace(0, len(offsets)-1, len(offsets))
    p = np.polyfit(xoff[~np.isnan(offsets)], offsets[~np.isnan(offsets)], deg=3)
    
    # Subtract the fit
    diff = np.asarray([p[0]*xoff**3 + p[1]*xoff**2 + p[2]*xoff + p[3]]*int(array_shape[0] - loc))

    image[int(loc):,:] += diff

    if return_ref:
        ref[int(loc):,:] += diff
        return image, ref
    else:
        return image

def destriping(image, ref_row=None):

    """
    Correct horizontal striping.
    
    Parameters:
    ----------------------
    image: 2D numpy array
        Input image.
    ref_row (optional): int
        Reference row to use. If None, the median of all rows is used.
        
    Returns:
    ----------------------    
    image: 2D numpy array
        Corrected image.
    """
        
    # Compute the offsets between the image rows
    offsets = np.nanmedian(image, axis=1)
    
    if ref_row is None:
        # If no reference, use the median of the rows as the zero-point reference
        offsets -= np.nanmedian(offsets)
    else:
        # Use the reference as the zero-point
        offsets -= offsets[ref_row]

    # Subtract the offsets
    return np.asarray([offsets]*np.shape(image)[1]).T

def Circular_Gaussian2D(xy, amp, sigma, offset, x0, y0, ravel=True):
    '''
    2D Circular Gaussian function
    '''
    x, y = xy
    
    model = (offset + amp*np.exp(-1*((x-x0)**2 + (y-y0)**2)/sigma))
    
    if ravel == True:
        return model.ravel()
    else:
        return model
        
def Gaussian2D(xy, amp, sigmax, sigmay, offset, x0, y0, ravel=True):
    '''
    2D Gaussian function, raveled
    '''
    x, y = xy

    model = (offset + amp*np.exp(-1*((x-x0)**2/sigmax + (y-y0)**2/sigmay)))
    
    if ravel == True:
        return model.ravel()
    else:
        return model

def airy_disk(xy, amp, sigmax, sigmay, offset, p, x0, y0, e=0.11, ravel=True):

    '''
    Airy disk function with obstruction/rotation
    '''
    x, y  = xy

    rad = np.pi*np.sqrt((((x-x0)*np.cos(p) + (y-y0)*np.sin(p))/sigmax)**2 +
                        (((x-x0)*np.sin(p) - (y-y0)*np.cos(p))/sigmay)**2)

    model = (amp/(1-e**2)**2) * (2*j1(rad)/rad - 2*e*j1(e*rad)/rad)**2 + offset

    model[rad == 0] =  amp/(1-e**2)**2

    if ravel == True:
        return model.ravel()
    else:
        return model

def modified_airy_disk(xy, amp, wavelength, aperture, axis_ratio, offset, p, x0, y0, e=0.11):
    
    '''
    Airy disk function with obstruction, rotation, and wavelength dependence
    '''
    x, y  = xy

    # If operating at a single wavelength, compute without numerical integration
    if wavelength.ndim == 0:

        # Calculate input radius into bessel function, accounting for image scale (0.0179"/px)
        rad = (np.pi*np.sqrt( (((x-x0)*np.cos(p) + (y-y0)*np.sin(p)))**2 +
               (((x-x0)*np.sin(p) - (y-y0)*np.cos(p))*axis_ratio)**2 ) * (aperture * 0.0179/206264.8) /
               (wavelength * 10**(-6)))

        model = (2*j1(rad)/rad - 2*e*j1(e*rad)/rad)**2
        
        model[rad == 0] =  1

        model = model*(amp/(1-e**2)**2)

        model += offset

    else:

        # Calculate input radius into bessel function, accounting for image scale (0.0179"/px)
        rad = np.einsum('i,jk->ijk', (1 / (wavelength * 10**(-6))),
                        np.pi*np.sqrt((((x-x0)*np.cos(p) + (y-y0)*np.sin(p)))**2 +
                                      (((x-x0)*np.sin(p) - (y-y0)*np.cos(p))*axis_ratio)**2) *
                                      (aperture * 0.0179/206264.8))   

        model = (2*j1(rad)/rad - 2*e*j1(e*rad)/rad)**2
        
        model[rad == 0] =  1
        
        model =  np.einsum('i,ijk->ijk', (amp/(1-e**2)**2), model)

        model += offset

    return model

def sinc_gauss(xy, amp, sigma, phase, x0, y0, r0, offset, power):

    '''
    Models trefoil in a single airy ring using sine and gaussian functions.
    '''

    x, y  = xy

    rad = np.sqrt((x-x0)**2 + (y-y0)**2)

    model = (np.sin(phase + 3*np.arctan2((x-x0),
                   (y-y0)))**power*amp*np.exp(-1*((rad-r0)/sigma)**2) +
             offset)
    
    return model

def center_triangle(xy, amp, amp2, sigma, sigma2, phase, x0, y0, r0, r02, ravel=True):
    
    '''
    Models trefoil in the zeroth and first maxima in the airy pattern.
    '''
    
    model =  (sinc_gauss(xy, amp, sigma, phase, x0, y0, r0, 0, 1) +
                sinc_gauss(xy, amp2, sigma2, phase + np.pi, x0, y0, r02, 0, 1))
    
    if ravel == True:
        return model.ravel()
    else:
        return model

def empirical_psf_fit(cutout, wvl_interp, relative_flux, model_trefoil=True):

    """
    Fit an empirically derived PSF to an image cutout.

    Parameters:
    ----------------------
    cutout: 2D numpy array
        Input image.
    wvl_interp: 1D numpy array
        An array of wavelengths. 
    relative_flux: 1D numpy array
        Relative flux for each respective wavelength in wvl_interp.
    model_trefoil (optional): boolean
        Enables the modeling of trefoil in the PSF. Enabled by default.

    Returns:
    ----------------------    
    reffit: 1D numpy array
        Airy disk fitting parameters (floats):
        amp:
            Peak amplitude of the airy disk model.
        sigmax:
            Standard deviation in the x-axis.
        sigmay:
            Standard deviation in the y-axis.
        offset:
            Additive offset/intercept.
        p:
            Rotation angle.
        x0:
            Horizontal coordinate of the origin.
        y0:
            Vertical coordinate of the origin.

    lbtfit: 1D numpy array
        Empirical psf fitting parameters (floats):
        amp:
            Peak amplitude of the airy disk model.
        offset:
            Additive offset/intercept.
        aperture:
            Baseline of the airy disk model in the x-axis.
        axis_ratio:
            Ratio of the standard deviations between the
            y-axis and x-axis.

    trifit: 1D numpy array
        Trefoil fit parameters:
        amp:
            Peak amplitude of the central trefoil pattern.
        amp2:
            Peak amplitude of the outer trefoil pattern.
        sigma:
            Standard deviation of the central trefoil pattern in
            the radial direction.
        sigma2:
            Standard deviation of the radial trefoil pattern in
            the radial direction.
        phase:
            Phase angle of the trefoil pattern.
        r0:
            Radial location of the central trefoil pattern.
        r02:
            Radial location of the outer trefoil pattern.
    """
    
    cutout_shape = np.shape(cutout)

    wx = np.linspace(0, cutout_shape[0]-1, cutout_shape[0])
    wy = np.linspace(0, cutout_shape[1]-1, cutout_shape[1])
    wx, wy = np.meshgrid(wx, wy)
        
    # Run curve_fit to get airy best fit parameters
    reffit, _ = curve_fit(airy_disk, (wx, wy), cutout.ravel(), p0=[np.max(cutout), 30,30, -1, 0,
                          0.5*cutout_shape[0], 0.5*cutout_shape[1]],
                          bounds=([0, 1, 1, 1*-np.inf, 0, 1, 1],
                                  [10*np.max(cutout), 200, 200, np.inf, 2*np.pi,
                                   cutout_shape[0], cutout_shape[1]]))

    # Create wrapper for modified_airy_disk function, input best fits from the airy_disk function
    def mod_wrapper(xy, amp, offset, aperture, axis_ratio, ravel=True):

        model = (amp*np.mean(modified_airy_disk(xy, relative_flux, wvl_interp, aperture,
                                                axis_ratio, 0, reffit[4], reffit[5],
                                                reffit[6]), axis=0) +
                 offset)
        
        if ravel == True:
            return model.ravel()
        else:
            return model

    # Run curve_fit using the wrapper to get modified airy best fit parameters
    lbtfit, _ = curve_fit(mod_wrapper, (wx, wy), cutout.ravel(), p0=[reffit[0], reffit[3], 8, 1],
                          bounds=([0, 1*-np.inf,6, .5], [10*np.max(cutout), np.inf, 10, 2]))  

    if model_trefoil:

        # Create PSF model using best fit parameters
        lbt_model = mod_wrapper((wx,wy), lbtfit[0], lbtfit[1], lbtfit[2], lbtfit[3], ravel=False)

        # Create wrapper for center_triangle, input best fits from the airy_disk function
        def tri_wrapper(xy, amp, amp2, sigma, sigma2, phase, r0, r02):

            return center_triangle(xy, amp, amp2, sigma, sigma2, phase, reffit[5], reffit[6], r0, r02)

        # Calculate model residual
        residual = cutout - lbt_model
            
        # Run curve_fit to get best fit parameters of trefoil from the residual
        trifit, _ = curve_fit(tri_wrapper, (wx, wy), residual.ravel(),
                              p0=[1e-4, 1e-4, 9, 9, 5.2, 14.38, 14.38*np.sqrt(3)],
                              bounds=([0, 0, 1, 1, 0, 1, 1],
                                      [10*np.max(cutout),10*np.max(cutout),
                                       20, 20, 2*np.pi, 80, 80]),
                              maxfev=10**5)
        
        return reffit, lbtfit, trifit
        
    else:
        
        return reffit, lbtfit, []

def pad_frame(frame, px, py, padding):
    
    """
    Pad frame.

    Parameters:
    ----------------------
    frame: 2D numpy array
        Input image.
    px: 1D numpy array
        Array enumerating columns of the alignment grid
    py: 1D numpy array
        Array enumerating rows of the alignment grid
    padding: integer tuple
        Tuple describing number of nan columns and rows to
        add to the image.

    Returns:
    ----------------------    
    image: 2D numpy array
        Padded image.
    """

    # Padding frame dependent on the signs of the padding tuple
    # There are four different cases for how to concatenate the nans
    if padding[1] < 0:
        
        frame = np.concatenate((frame, np.nan*np.ones((-1*padding[1], px - np.abs(padding[0])))))
        
    else:
        
        frame = np.concatenate((np.nan*np.ones((padding[1], px - np.abs(padding[0]))), frame))

    if padding[0] < 0:
        
        frame = np.concatenate((frame, np.nan*np.ones((py, -1*padding[0]))), axis=1)

    else:

        frame = np.concatenate((np.nan*np.ones((py, padding[0])), frame), axis=1)

    return frame
    
def align_frame(frame, px, py, padding, offset, method="cubic"):
    
    """
    Align frame using computed offset/alignment grid.

    Parameters:
    ----------------------
    frame: 2D numpy array
        Input image.
    px: 1D numpy array
        Array enumerating columns of the alignment grid
    py: 1D numpy array
        Array enumerating rows of the alignment grid
    padding: integer tuple
        Tuple describing number of nan columns and rows to add to the
        image.
    offset: float tuple
        Tuple encoding image offset from the alignment grid
    method (optional): string
        Interpolation method for scipy.interp.RegularGridInterpolator,
        default is cubic interpolation. Linear interpolation is much
        faster but imprecise especially at the center of the PSF.
    
    Returns:
    ----------------------    
    image: 2D numpy array
        Aligned image.
    """

    # Pad frame to fit to the alignment grid
    frame = pad_frame(frame, len(px), len(py), padding)

    # Set all nans to the unbiased mean value if using a nonlinear interpolation
    if method != "linear":
    
       # Find a suitable unbiased mean value
        val = np.nanmean(frame[(frame < np.nanmedian(frame) + np.nanstd(frame)) &
                               (frame > np.nanmedian(frame) - np.nanstd(frame))])
        
        frame[np.isnan(frame)] = val

    # Construct the interpolator
    interp = RegularGridInterpolator((py, px), frame, method=method, bounds_error=False)
    Y, X = np.meshgrid(px+offset[0], py+offset[1])

    # Interpolate the image to the grid
    image = interp((X, Y))

    return image

def calculate_expected_flux(stellar_temp, nomic_filter = "Nprime",
                                 min_cutoff=0.05, n_interp=30):
    
    """
    Calculates expected relative flux density for a star based on
    stellar temperature, atmospheric transmission, and filter
    transmission.

    Parameters:
    ----------------------
    stellar_temp: float
        Temperature of the star in Kelvins.
    nomic_filter (optional): string
        Wavelengths at which to calculate stellar flux density. 
        Default filter is "Nprime".
    min_cutoff (optional): float
        Minimum transmission for designating filter cutoff. 
        Default value is 0.05.
    n_interp (optional): integer
        Number of points to use for interpolating filter/transmission
        curves. Default value is 30 points.

    Returns:
    ----------------------    
    wvl_interp: 1D numpy array
        An array of wavelengths. 
    relative_flux: 1D numpy array
        Relative flux for each respective wavelength in wvl_interp.
    """
    
    # Load atmospheric transmission
    transmit = np.load("filterdata/MtGrahamTransmittance.npz")
    
    # Create interpolator for the atmospheric transmission
    atmo_transmit_interp = interp1d(transmit["arr_0"], transmit["arr_1"], kind="linear",
                                    bounds_error=False, fill_value="extrapolate")

    # Open N prime filter curve
    if nomic_filter == "Nprime":
        
        t = QTable.read(r"filterdata/Nprime_filter_curve.txt", format="ascii")
        wavelength = np.asarray(t[t.keys()[0]])
        filter_transmission = np.asarray(t[t.keys()[1]])
        
    # Open narrowband filter table
    else:
        t = QTable.read(r"filterdata/JDSUNarrowbandFilters.csv")
        wavelength = np.asarray(t[nomic_filter+" [um]"])
        # Convert percentages to decimals
        filter_transmission = np.asarray(t[nomic_filter+" %T"])/100

    # Create interpolator for filter transmission curve
    filterinterp = interp1d(wavelength, filter_transmission, kind="linear",
                            bounds_error=False, fill_value="extrapolate")

    # Create wavelength list based on filter cutoffs
    wvl_interp = np.linspace(np.min(wavelength[filter_transmission > min_cutoff]),
                             np.max(wavelength[filter_transmission > min_cutoff]), n_interp)

    # Calculate stellar flux density based on stellar temperature
    stellar_flux = np.log10(stellar_flux_density(wvl_interp, 1, 1, stellar_temp))

    # Calculate relative spectral flux density
    relative_flux = (atmo_transmit_interp(wvl_interp) * filterinterp(wvl_interp) * stellar_flux / 
                     np.max(stellar_flux))

    return wvl_interp, relative_flux

def stellar_flux_density(wavelengths, distance, stellar_radius, stellar_temp):
    
    """
    Align frame using computed offset/alignment grid.

    Parameters:
    ----------------------
    wavelengths: 1D numpy array
        Wavelengths at which to calculate stellar flux density.
    distance: float
        Distance of the star in parsecs.
    stellar_radius: float
        Radius of the star in solar radii.
    stellar_temp: float
        Temperature of the star in Kelvins.

    Returns:
    ----------------------    
    spectral_flux_density: 1D numpy array
        Calculated spectral flux density for each input wavelength.
    """
    
    # Get frequency in dimensionless units (h*c/kT * lambda)
    freq = (c.h * c.c / (c.k_B*stellar_temp*u.K*(wavelengths)*u.micron))
    
    # Calculate expected spectral flux density based on stellar properties
    spectral_flux_density = ((((2 * np.pi * c.c * (stellar_radius*c.R_sun / (distance*u.pc))**2) /
                            ((wavelengths*u.micron)**4)) / (np.exp(freq) - 1))\
                            .to(1 / (u.s*u.m**2*u.micron)).value)
    
    '''
    irradiance = (((2 * np.pi * c.h * (c.c*stellar_radius*c.R_sun /(1*u.AU))**2) /
                 ((w1*u.micron)**5))/(np.exp(freq) - 1)).to(u.W/(u.m**2*u.nm))
    '''
    
    return spectral_flux_density
    
