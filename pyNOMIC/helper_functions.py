#----------------------------------------
# IMPORTS
#----------------------------------------

from astropy.io import fits
import numpy as np
from tqdm.auto import tqdm
from astropy.convolution import convolve, Gaussian1DKernel
from scipy.special import j1
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
    
        # Find desired array shape based on the spatial bin required, and the requirement of even image sizes
        array_shape = (shape[0], (2*np.ceil(shape[1]/(2*spatial_bin))).astype(np.int32),\
                      (2*np.ceil(shape[2]/(2*spatial_bin))).astype(np.int32)) 
        
        # Pad image cube in preparation for binning to the array shape
        new_cube = np.pad(img_cube, ((0,0),(int(0.5*(spatial_bin*array_shape[1]-shape[1])),\
                                int(0.5*(spatial_bin*array_shape[1]-shape[1]))),\
                    (int(0.5*(spatial_bin*array_shape[2]-shape[2])),\
                     int(0.5*(spatial_bin*array_shape[2]-shape[2])))), constant_values=np.nan)
        
        #new_cube[new_cube == flag_value] = np.nan
    
        # Perform binning with np.sum and np.reshape
        new_cube = np.sum(np.sum(np.reshape(new_cube, (array_shape[0], array_shape[1],\
                    spatial_bin, array_shape[2], spatial_bin)), axis=4), axis=2)

    else:
        
        new_cube = img_cube 
        array_shape = np.shape(img_cube)
        
    return new_cube, array_shape


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
    distance = (np.sqrt((X - center[0])**2 + (Y-center[1])**2) - inner_radius)/outer_radius

    # Set all values inside outer radius to 0, set all values outside the radius to 1
    distance[distance < 0] = 0
    distance[distance > 1] = 1

    # Reverse image mask
    return 1 - distance


def repairChannelEdges(image, loc):
        
    """
    Fill in data for three horizontal channel edges on the NOMIC detector, to prevent artifacting. Each channel edge consists of three row and requires data for each.

    Parameters:
    ----------------------
    image: 2D numpy array
        Input image.
    loc: int
        Location of the central row of the channel edge to be corrected.
        
    Returns:
    ----------------------    
    image: 2D numpy array
        Corrected image.
    """
    
    gauss = Gaussian1DKernel(stddev=5)
    
    # Replicate noise profile of adjacent rows
    image[loc+1,:] *= 0.7*np.std(image[loc+2,:])/np.std(image[loc+1,:])
    image[loc-1,:] *= 0.7*np.std(image[loc-2,:])/np.std(image[loc-1,:])
    # For middle row of the channel edge, average the top and bottom
    image[loc,:] *= 0.5*(np.std(image[loc+1,:]) + np.std(image[loc-1,:]))/np.std(image[loc,:])

    # Get smoothed difference between channel edge and adjacent row and add it back
    image[loc+1,:] +=  convolve(image[loc+2,:]-image[loc+1,:], gauss)
    image[loc-1,:] +=  convolve(image[loc-2,:]-image[loc-1,:], gauss)

    # Get smoothed difference between middle row and averaged top and bottom rows and add it back
    image[loc,:] += convolve( 0.5*(image[loc-1,:] + image[loc+1,:])-image[loc,:], gauss)

    return image


def repairVerticalLine(image, loc):
        
    """
    Repair vertical line artifact in the NOMIC detector while preserving the unique noise profile.

    Parameters:
    ----------------------
    image: 2D numpy array
        Input image.
    loc: int
        Location of the column to be corrected.
        
    Returns:
    ----------------------    
    image: 2D numpy array
        Corrected image.
    """
    
    gauss = Gaussian1DKernel(stddev=5)
    
    # Average adjacent columns
    true = 0.5*(image[:, loc+1] + image[:, loc-1])

    # Get smoothed difference between the column and the averaged adjacent columns and add it back
    image[:, loc] += convolve(true - image[:, loc], gauss)

    return image


def repairHorizontalLine(image, loc):
        
    """
    Repair horizontal line artifact in the NOMIC detector while preserving the unique noise profile.

    Parameters:
    ----------------------
    image: 2D numpy array
        Input image.
    loc: int
        Location of the row to be corrected.
        
    Returns:
    ----------------------    
    image: 2D numpy array
        Corrected image.
    """
    
    gauss = Gaussian1DKernel(stddev=5)
    
    # Average adjacent columns
    true = 0.5*(image[loc+1,:] + image[loc-1,:])

    # Get smoothed difference between the column and the averaged adjacent columns and add it back
    image[loc,:] += convolve(true - image[loc,:], gauss)

    return image


def repairVerticalBias(image, loc):
    
    """
    Offset bias between two sides of the detector, column-wise.

    Parameters:
    ----------------------
    image: 2D numpy array
        Input image.
    loc: int
        Location of the column separating the two sides of the detector.
        
    Returns:
    ----------------------    
    image: 2D numpy array
        Corrected image.
    """
    
    array_shape = np.shape(image)

    # Calculate offsets between the two sides solely based on the 5 adjacent columns on either side
    offsets = np.mean(image[:,int(loc - 6):int(loc - 1)] -\
                      image[:,int(loc):int(loc + 5)], axis=1)

    # Construct a polynomial fit along the rows to model the offset
    xoff = np.linspace(0, len(offsets)-1, len(offsets))
    p = np.polyfit(xoff, offsets, deg=3)

    # Subtract the fit
    image[:, int(loc):] =  image[:, int(loc):] +\
            np.asarray([p[0]*xoff**3 + p[1]*xoff**2 + p[2]*xoff + p[3]]*int(array_shape[1] - loc)).T

    return image
    
def repairHorizontalBias(image, loc):
    
    """
    Offset bias between two sides of the detector, row-wise.

    Parameters:
    ----------------------
    image: 2D numpy array
        Input image.
    loc: int
        Location of the row separating the two sides of the detector.
        
    Returns:
    ----------------------    
    image: 2D numpy array
        Corrected image.
    """
    
    array_shape = np.shape(image)
    
    # Calculate offsets between the two sides solely based on the 5 adjacent rows on either side
    offsets = np.mean(image[int(loc - 6):int(loc - 1),:] -\
                      image[int(loc):int(loc + 5),:], axis=0)
    
    # Construct a polynomial fit along the columns to model the offset
    xoff = np.linspace(0, len(offsets)-1, len(offsets))
    p = np.polyfit(xoff, offsets, deg=3)
    
    # Subtract the fit
    image[int(loc):,:] = image[int(loc):,:] +\
            np.asarray([p[0]*xoff**3 + p[1]*xoff**2 + p[2]*xoff + p[3]]*int(array_shape[0] - loc))

    return image

def destriping(image, ref=None):

    """
    Correct horizontal striping.
    
    Parameters:
    ----------------------
    image: 2D numpy array
        Input image.
    ref (optional): int
        Reference row to use. If None, the median of all rows is used.
        
    Returns:
    ----------------------    
    image: 2D numpy array
        Corrected image.
    """

    # Compute the offsets between the image rows
    offsets = np.nanmedian(image, axis=1)
    if ref is None:
        # If no reference, use the median of the rows as the zero-point reference
        offsets -= np.nanmedian(offsets)
    else:
        # Use the reference as the zero-point
        offsets -= offsets[ref]

    # Subtract the offsets
    return image - np.asarray([offsets]*np.shape(image)[1]).T

def Circular_Gaussian2D_ravel(xy, amp, sigma, offset, x0, y0):
    '''
    2D Circular Gaussian function, raveled
    '''
    x, y = xy
    return (offset + amp*np.exp(-1*((x-x0)**2 + (y-y0)**2)/sigma)).ravel()

def Circular_Gaussian2D(xy, amp, sigma, offset, x0, y0):
    '''
    2D Circular Gaussian function
    '''
    x, y = xy
    return (offset + amp*np.exp(-1*((x-x0)**2 + (y-y0)**2)/sigma))

def Gaussian2D_ravel(xy, amp, sigmax, sigmay, offset, x0, y0):
    '''
    2D Gaussian function, raveled
    '''
    x, y = xy
    return (offset + amp*np.exp(-1*((x-x0)**2/sigmax + (y-y0)**2/sigmay))).ravel()

def Gaussian2D(xy, amp, sigmax, sigmay, offset, x0, y0):
    '''
    2D Gaussian function
    '''
    x, y = xy
    return (offset + amp*np.exp(-1*((x-x0)**2/sigmax + (y-y0)**2/sigmay)))

def airydisk_ravel(xy, amp, rx, ry, offset, p, x0, y0, e=0.11):
    '''
    Airy disk function with obstruction/rotation, raveled
    '''
    x, y  = xy

    rad = np.pi*np.sqrt( (((x-x0)*np.cos(p) + (y-y0)*np.sin(p))/rx)**2 + (((x-x0)*np.sin(p) - (y-y0)*np.cos(p))/ry)**2)

    model = (amp/(1-e**2)**2) * (2*j1(rad)/rad - 2*e*j1(e*rad)/rad)**2 + offset

    model[rad == 0] =  amp/(1-e**2)**2
    
    return model.ravel()
    
def airydisk(xy, amp, rx, ry, offset, p, x0, y0, e=0.11):

    '''
    Airy disk function with obstruction/rotation
    '''
    x, y  = xy

    rad = np.pi*np.sqrt( (((x-x0)*np.cos(p) + (y-y0)*np.sin(p))/rx)**2 + (((x-x0)*np.sin(p) - (y-y0)*np.cos(p))/ry)**2 )

    model = (amp/(1-e**2)**2) * (2*j1(rad)/rad - 2*e*j1(e*rad)/rad)**2 + offset

    model[rad == 0] =  amp/(1-e**2)**2
    
    return model

def circular_mask(center, radius, width, height):
    '''
    Creates circular mask of certain radius in an image of certain width and height
    '''
    Y, X = np.ogrid[:height, :width]
    distance = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = distance <= radius
    
    return mask

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
        Tuple describing number of nan columns and rows to add to the image.

    Returns:
    ----------------------    
    image: 2D numpy array
        Padded image.
    """

    # Padding frame dependent on the signs of the padding tuple, with four different cases for how to concatenate the nans
    if padding[1] < 0:
        
        frame = np.concatenate((frame, np.nan*np.ones((-1*padding[1], px - np.abs(padding[0])))))
        
    else:
        
        frame = np.concatenate((np.nan*np.ones((padding[1], px - np.abs(padding[0]))), frame))

    if padding[0] < 0:
        
        frame = np.concatenate((frame, np.nan*np.ones((py, -1*padding[0]))), axis=1)

    else:

        frame = np.concatenate((np.nan*np.ones((py, padding[0])), frame), axis=1)

    return frame
    
def align_frame(frame, px, py, padding, offset, method="linear"):
    
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
        Tuple describing number of nan columns and rows to add to the image.
    offset: float tuple
        Tuple encoding image offset from the alignment grid
    method (optional): string
        Interpolation method for scipy.interp.RegularGridInterpolator, default is cubic interpolation.
        Linear interpolation is much faster but imprecise especially at the center of the PSF.
    
    Returns:
    ----------------------    
    image: 2D numpy array
        Aligned image.
    """

    # Pad frame to fit to the alignment grid
    frame = pad_frame(frame, len(px), len(py), padding)

    # Find the maximum value
    val = np.nanmax(frame)

    # Set all nans to 1e9 times the maximum value if using a nonlinear interpolation
    if method != "linear":
        frame[np.isnan(frame)] = 1e9*val

    # Construct the interpolator
    interp = RegularGridInterpolator((py, px), frame, method=method, bounds_error=False)
    Y, X = np.meshgrid(px+offset[0], py+offset[1])

    # Interpolate the image to the grid
    image = interp((X, Y))

    # Set all values sufficiently above the maximum value to nans if using a nonlinear interpolation
    if method != "linear":
        image[image > val*1.1] = np.nan
    
    return image
