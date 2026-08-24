#----------------------------------------
# IMPORTS
#----------------------------------------
import numpy as np
import os, pathlib
from tqdm.auto import tqdm
from astropy.io import fits
from astropy.convolution import convolve_fft, Ring2DKernel
from scipy.stats import linregress
from multiprocessing.pool import ThreadPool as Pool
from vip_hci.psfsub.utils_pca import pca_annulus
from vip_hci.psfsub.pca_local import pca_annular
import pyNOMIC.helper_functions as hf

#----------------------------------------
# CLASSES
#----------------------------------------

class InjectSource(object):
    
    '''
    Inject simulated sources into data.
    '''

    def __init__(self, params):

        """
        Parameters (contained inside a tuple):
        ----------------------
        frames: List of raw file paths or 3D image array
        directory: string/Path object
            Directory where images will be saved
        wx: 1D numpy array
            Array enumerating columns of the alignment grid
        wy: 1D numpy array
            Array enumerating rows of the alignment grid
        array_shape: integer tuple
            Tuple containing image dimensions, from numpy.shape
        injection_radius: float
            Radius from the stellar psf in which to inject a source.
        injection_angle: float
            Position angle at which to inject a source.
        contrast: float
            Contrast ratio between the injection and star.
        width: integer
            Width of the output image.
        height: integer
            Height of the output image.
        highpassrad: integer
            Highpass filter radius. None by default, disabling highpass
            filtering.
        pxscale: float
            The image scale of the images in arcseconds per pixel.
            Default is 0.0179"/px, for LBTI-NOMIC.
        """
        
        self.params = params
    
    def __call__(self, star_pars):

        """
        Parameters (contained inside a tuple):
        ----------------------
        frame_indices: 1D numpy array
            Indices along the image array
        reffits: 1D numpy array
            Array of airy disk fitting parameters corresponding to
            each file in frames
        para_angles: list or array
            List of parallactic angles corresponding to each file
            in framesv
        psf_locs (optional): 2 x len(frames) numpy array
            Array containing pixel coordinates of psf locations
            in the input images. If None, the center of the image
            is assumed. 
        """

        (frames, directory, wx, wy, array_shape,
         injection_radius, injection_angle,
         contrast, width, height, highpassrad, pxscale) = self.params

        reffit = np.zeros(7)
        psf_loc = np.zeros(2)
        (single_frame, reffit[0], reffit[1], reffit[2], reffit[3], reffit[4],
         reffit[5], reffit[6], para_angle, psf_loc[0], psf_loc[1]) = star_pars

        # Create source using the airy disk model from the star
        source = hf.airy_disk((wx, wy), contrast*reffit[0], reffit[1], reffit[2], 0, reffit[4],
                              psf_loc[1] + injection_radius*np.cos((injection_angle -
                                                                     para_angle)*np.pi/180),
                              psf_loc[0] + injection_radius*np.sin((injection_angle -
                                                                     para_angle)*np.pi/180),
                              ravel=False)

        # Load image, depending on whether frames are in files or memory
        if frames is not None:
    
            img = frames[int(single_frame)][width[0]:width[1], height[0]:height[1]] + source
    
        else:
            
            hdul = fits.open(single_frame)
            img = hdul[0].data[width[0]:width[1], height[0]:height[1]] + source
            hdul.close()

        if highpassrad is not None:
            
            fwhm = 2*np.sqrt(2*np.log(2)*(reffit[1]+reffit[2]))*pxscale

            img = hf.simple_highpass(img, psf_loc, array_shape,
                                     highpassrad, fwhm)
            
        if directory is not None:
            # Save image to file
            newhdul = fits.HDUList([fits.PrimaryHDU(data=(img))])      
            newhdul.writeto(os.path.join(directory, "injected_"+single_frame.name), overwrite=True)
            newhdul.close()
    
            return True, True
    
        else:
    
            return img, single_frame


class ForwardModel(object):

    '''
    Performs forward modeling on an image cube at a certain position
    to measure contrast.
    '''

    def __init__(self, params):

        """
        Parameters (contained inside a tuple):
        ----------------------
        frames: List of file paths or 3D image array
        para_angles: list or array
            List of parallactic angles corresponding to each file
            in frames
        reffits: 1D numpy array
            Array of airy disk fitting parameters corresponding to
            each file in frames
        array_shape: integer tuple
            Tuple containing image dimensions, from numpy.shape
        curve_prior: 1D numpy array
            A contrast curve used as a starting assumption for performing
            SNR measurements.
        aperture_radius: integer
            Radius of SNR measurement apertures in pixels.
        ncomp: integer
            Number of KLIP components.
        delta_rot: float
            Excludes images with angles within delta_rot from
            the PSF library for each image. Default is 0, which allows
            for the PSF library to be only computed once.
        n_segments: integer
            Number of segments in each annulus for PCA.
        highpassrad: integer
            Highpass filter radius. None by default, disabling highpass
            filtering.
        tolerance: float
            Allowable tolerance for computing the SNR as a percent.
        high_buffer: float
            Multiplicative factor applied to curve_prior to ensure that
            modelling starts with a high SNR source.
        max_iterations: integer
            Maximum number of iterations allowed when recursively
            calculating source SNR.
        add_info: list
            Additional information to include in error files.
        """
        
        self.params = params
    
    def __call__(self, coords):

        """
        Parameters (contained inside coords):
        ----------------------
        injection_radius: 1D numpy array
            Radius from the stellar psf in which to inject a source.
        injection_angle: 1D numpy array
            Position angle at which to inject a source
        Returns: 
        ---------------------- 
        last_contrast: float
            Computed contrast limit.
        curr_snr: float
            SNR of the injected source with contrast equal
            to last_contrast.
        """
        
        (frames, para_angles, reffits, array_shape, curve_prior,
         aperture_radius, ncomp, delta_rot, n_segments, highpassrad,
         tolerance, high_buffer, max_iterations, add_info, nproc) = self.params                    

        # Initialize using buffer and assumed prior for contrast curve
        last_contrast = high_buffer*curve_prior(coords[0])

        snr_plot = np.array([])
        contrast_plot = np.array([])

        # Initialize error flags
        anticorrelation = False
        correlation = False
        lin_error = False
        counter = 0
    
        while True:
         
            try:
                
                (injected_cube,
                 injected_shape) = inject_source(frames, para_angles, reffits, array_shape,
                                                 contrast=last_contrast,
                                                 injection_radius=coords[0],
                                                 injection_angle=coords[1],
                                                 crop_size=2*int(np.ceil(1.05*(coords[0]+
                                                                               aperture_radius))),
                                                 highpassrad=highpassrad, save_files=False,
                                                 threadcount=nproc)

                # Get measured SNR with assumed contrast
                (curr_snr,
                 curr_noise_factor) = measure_snr(injected_cube, para_angles, aperture_radius,
                                                  coords[0], coords[1], ncomp=ncomp,
                                                  delta_rot=delta_rot, n_segments=n_segments,
                                                  nproc=nproc)

                # Add SNRs and contrasts to plot
                snr_plot = np.append(snr_plot, curr_snr)
                contrast_plot = np.append(contrast_plot, last_contrast)
                #print("Contrast: ",last_contrast)
                #print("SNR: ",curr_snr)

                # If SNR is within tolerance, break loop
                if (np.abs(curr_snr - 5)/(5) < tolerance):
                    
                    break
                    
                else:

                    # Check to make sure SNR is higher than 5 (prefer to converge from high SNR)
                    if np.max(snr_plot) > 5:
                        
                        if (len(snr_plot) > 1):

                            # Create a line between the last two points and extrapolate
                            results = linregress(snr_plot[-2:], np.log(contrast_plot[-2:]))
                            curr_approx_contrast = np.exp(np.round(results[0]*5 + results[1], 3)) 
                            #print("Linregress: ", curr_approx_contrast)

                            # If contrast is not within expected limits
                            if curr_approx_contrast < 1e-6 or curr_approx_contrast > 1e-1:

                                # If SNR is negative, find the closest SNR to 5
                                if curr_snr < 0:
                                    best_snr = snr_plot[1:][np.nanargmin(np.abs(snr_plot[1:] - 5))]
                                    best_contrast = contrast_plot[1:]\
                                                           [np.nanargmin(np.abs(snr_plot[1:] - 5))]
                                    snr_plot = snr_plot[:-1]
                                    contrast_plot = contrast_plot[:-1]
                                else:
                                    best_snr = curr_snr
                                    best_contrast = last_contrast

                                # Recalculate contrast using either the closest or the last SNR
                                curr_approx_contrast = np.abs((5/best_snr)*(best_contrast))

                                '''
                                If the recalculated contrast has already been used, recalculate
                                instead by linear regression of the whole non-negative dataset
                                '''
                                if np.isin(curr_approx_contrast, contrast_plot):                                        
                                    results = linregress(snr_plot[snr_plot > 0],
                                                         np.log(contrast_plot[snr_plot > 0]))
                                    curr_approx_contrast = np.exp(np.round(results[0]*5 +
                                                                           results[1], 3)) 
                                    #print("LR Linregress: ", curr_approx_contrast)
                                #print("LR Scaling: ", curr_approx_contrast)
                            '''
                            plt.figure(1)
                            plt.scatter(np.asarray(snr_plot), np.asarray(contrast_plot),
                                        s=10, color="orange")
                            plt.plot(snr_plot[-2:], np.log(contrast_plot[-2:]))
                            plt.scatter([5],[curr_approx_contrast],  s=20, color="red")
                            plt.vlines([5], ymin=1e-4, ymax=8e-4, color="green")
                            plt.yscale("log")
                            plt.xlabel("SNR")
                            plt.ylabel("Contrast")
                            plt.show()
                            '''
                        else:
                            # If first data point, adjust based on proportion
                            curr_approx_contrast = np.abs((5/curr_snr)*(last_contrast))

                    else:
                        if len(snr_plot) > 1:
                            
                             # Check for anticorrelation and correlation
                            if (((snr_plot[-1] < snr_plot[-2]) &
                                 (contrast_plot[-1] > contrast_plot[-2])) or
                                ((snr_plot[-1] > snr_plot[-2]) &
                                 (contrast_plot[-1] < contrast_plot[-2]))):
                                
                                anticorrelation = True
                                curr_approx_contrast = 0.25*last_contrast

                            else:
                                correlation = True
                                curr_approx_contrast = 2*last_contrast
                                '''
                                if (((snr_plot[-1] > snr_plot[-2]) &
                                     (contrast_plot[-1] > contrast_plot[-2])) or
                                    ((snr_plot[-1] < snr_plot[-2]) &
                                     (contrast_plot[-1] < contrast_plot[-2]))):
                                '''
                        else:
                             # If SNR is too low, boost contrast by a factor of 2
                            curr_approx_contrast = 2*last_contrast
                            
                    last_contrast = curr_approx_contrast

                counter += 1
            
            except:

                # Log error if one is encountered
                lin_error=True
                np.savez("Error_log-"+"_radius-"+str(coords[0])+"_angle-"+str(coords[1])+
                         "_binning-"+str(add_info[0])+"_sigma-"+str(add_info[1])+"_ncomp-"+
                         str(ncomp)+"_highpass-"+ str(highpassrad)+".npz", coords, add_info[0],
                         add_info[1], ncomp, highpassrad)
                pass

            '''
            If both anticorrelation and correlation are encountered, or a linear fit error, or
            max iterations are exceeded, break the loop and try to save the closest SNR to 5
            '''
            if anticorrelation & correlation or counter > max_iterations or lin_error:
                
                try:
                    
                    last_contrast = contrast_plot[np.nanargmin(np.abs(snr_plot - 5))]
                    curr_snr = snr_plot[np.nanargmin(np.abs(snr_plot - 5))]
                    
                except:
                    print("Contrasts: ", contrast_plot)
                    print("SNRs: ", snr_plot)
                    last_contrast = np.nan
                    curr_snr = np.nan
                    
                if anticorrelation & correlation:
                    print("Absolute max reached: ", last_contrast)
                elif (counter > max_iterations):
                    print("Max iterations reached: ", last_contrast)
                else:
                    print("Linear fit error encountered.")
                break

        return last_contrast, curr_snr

class FluxSNR(object):

    def __init__(self, params):

        self.params = params
    
    def __call__(self, coords):

        image, array_shape, origin, aperture_radius = self.params

        x, y = coords
        x -= origin[0]
        y -= origin[1]
        source_radius = np.sqrt(x**2 + y**2)
        source_angle  = np.arctan2(y, x)

        if source_radius > aperture_radius:
            
            # Compute angles at which to create apertures
            circle_pos = np.linspace(0, 2*np.pi,
                                     int(np.floor(np.pi*source_radius/aperture_radius))+1)[:-1]

            fluxes = np.zeros(len(circle_pos))

            # Measure flux in each aperture by creating a mask
            for i in range(len(circle_pos)):
            
                mask = hf.circular_mask((origin[0] + source_radius*np.cos(circle_pos[i] +
                                                                          source_angle),
                                      origin[1] + source_radius*np.sin(circle_pos[i] +
                                                                          source_angle)),
                                     aperture_radius, array_shape[0], array_shape[1])

                fluxes[i] = np.nansum(image[mask])
            
            # Compute SNR
            noise_factor = (np.std(fluxes[1:])*np.sqrt(1 + 1/(len(fluxes) - 1)))
        
            snr = (fluxes[0] - np.mean(fluxes[1:]))/noise_factor
        
            return snr, noise_factor
            
        else:
            
            return np.nan, np.nan

#----------------------------------------
# FUNCTIONS
#----------------------------------------

def inject_source(frames, para_angles, reffits, array_shape,
                  injection_radius=80, injection_angle=45, contrast=0.01,
                  pxscale=0.0179, psf_locs=None, highpassrad=None,
                  crop_size=None, save_files=False, threadcount=50):

    """
    Injects a source into a cube of frames.
    
    Parameters (contained inside a tuple):
    ----------------------
    frames: List of file paths or 3D image array
    para_angles: list or array
        List of parallactic angles corresponding to each file
        in frames
    reffits: 1D numpy array
        Array of airy disk fitting parameters corresponding to
        each file in frames
    array_shape: integer tuple
        Tuple containing image dimensions, from numpy.shape
    injection_radius (optional): float
        Radius from the stellar psf in which to inject a source.
    injection_angle (optional): float
        Position angle at which to inject a source.
    contrast (optional): float
        Contrast ratio between the injection and star.
    psf_locs (optional): 2 x len(frames) numpy array
        Array containing pixel coordinates of psf locations
        in the input images. If None, the center of the image
        is assumed. 
    highpassrad (optional): integer
        Highpass filter radius. None by default, disabling highpass
        filtering.
    pxscale (optional): float
        The image scale of the images in arcseconds per pixel.
        Default is 0.0179"/px, for LBTI-NOMIC.
    crop_size (optional): integer
        Crops frames to image dimension of crop_size x crop_size. None
        by default, disabling cropping.
    save_files (optional): boolean
        Enables saving the injected files, False by default.
    threadcount (optional): integer
        Number of threads to employ in multithreading.
        Default value is 50 threads.       
        
    Returns: 
    ---------------------- 
    injected_files or injected_cube: List of file paths or 3D image array
    array_shape: integer tuple
        Tuple containing image dimensions, from numpy.shape
    """
    
    # Create model grid for the images, depending on if we are cropping
    if crop_size is None:
        
        x = np.linspace(0, array_shape[0]-1, array_shape[0])
        y = np.linspace(0, array_shape[1]-1, array_shape[1])
        width = (None, None)
        height = (None, None)
        
    else:
        x = np.linspace(0, crop_size-1, crop_size)
        y = np.copy(x)
        
        width = (int(0.5*(array_shape[0] - crop_size)), -1*int(0.5*(array_shape[0] - crop_size)))
        height = (int(0.5*(array_shape[1] - crop_size)), -1*int(0.5*(array_shape[1] - crop_size)))
        
        array_shape = (crop_size, crop_size)

    if psf_locs is None:
        psf_locs = np.asarray([[0.5*(array_shape[1] - 0), 0.5*(array_shape[0] - 1)]]*len(frames))
        
    wx, wy = np.meshgrid(y, x)

    # Check if frames is an image cube or list of files
    if frames.ndim == 3:
        frame_pars = np.vstack((np.arange(len(frames)), reffits[:,0], reffits[:,1], reffits[:,2],
                                reffits[:,3], reffits[:,4], reffits[:,5], reffits[:,6],
                                para_angles, psf_locs[:,0], psf_locs[:,1])).T

        first_arg = frames
    else:
        frame_pars = np.vstack((frames, reffits[:,0], reffits[:,1], reffits[:,2], reffits[:,3],
                                reffits[:,4], reffits[:,5], reffits[:,6], para_angles,
                                psf_locs[:,0], psf_locs[:,1])).T
        first_arg = None

    # Create directories if files are being saved
    if save_files:
            root_dir = os.path.dirname(os.path.dirname(frames[0]))
            injected_dir = os.path.join(root_dir, 'injected')
            if not os.path.exists(injected_dir):
                os.makedirs(injected_dir)
    else:
        injected_dir = None

    # Choosing between parallelized and non-parallelized methods
    if threadcount != 0:
        
        #if __name__ == "__main__":
        with Pool(threadcount) as pool:
                injected_cube, check_array = \
                    zip(*pool.imap(InjectSource((first_arg, injected_dir, wx, wy, array_shape,
                                                 injection_radius, injection_angle, contrast,
                                                 width, height, highpassrad, pxscale)),
                                   frame_pars))
        
        # Either save images or return in memory
        if save_files:
            
            injected_files = np.asarray(sorted(list(pathlib.Path(str(injected_dir))\
                .rglob('*.fits'))))
            injected_files = np.asarray([a for a in injected_files\
                                               if a.name[0]!='.'\
                                               and str(a.parent)==injected_dir])
        
            return injected_files, array_shape
            
        else:
                
            sorted_injections = np.asarray(injected_cube)[np.argsort(check_array)]
                
            return sorted_injections, array_shape
    
    else:

        source_gen = InjectSource((first_arg, injected_dir, wx, wy, array_shape,
                                   injection_radius, injection_angle, contrast,
                                   width, height, highpassrad))

        # Either save images or return in memory
        if save_files:        
        
            for i in range(len(frame_pars)):
                
                check, _ = source_gen(frame_pars[i])
                
            injected_files = np.asarray(sorted(list(pathlib.Path(str(injected_dir))\
                             .rglob('*.fits'))))
            injected_files = np.asarray([a for a in injected_files\
                                               if a.name[0]!='.'\
                                               and str(a.parent)==injected_dir])
            return injected_files, array_shape
            
        else:
                    
            injected_cube = np.zeros((len(frame_pars), array_shape[0], array_shape[1]))
            
            for i in range(len(frame_pars)):
                
                injected_cube[i], check = source_gen(frame_pars[i])

            return injected_cube, array_shape


def measure_snr(image_cube, para_angles, aperture_radius, source_radius, source_angle,
                ncomp=10, delta_rot=0, n_segments=1, nproc=None):
   
    """
    Measures the SNR of a source following Mawet et al. 2014.
    
    Parameters (contained inside a tuple):
    ----------------------
    image_cube: 3D image array
        Cube of frames
    para_angles: list or array
        List of parallactic angles corresponding to each frame
        in the cube.
    aperture_radius: integer
        Radius of SNR measurement apertures in pixels.
    source_radius: float
        Radius from the stellar psf where the source is located
    source_angle: float
        Position angle at which the source is located
    ncomp (optional): integer
        Number of KLIP components.
    delta_rot (optional): float
        Excludes images with angles within delta_rot from
        the PSF library for each image. Default is 0, which allows
        for the PSF library to be only computed once.
    n_segments (optional): integer
        Number of segments in each annulus for PCA.
    nproc (optional): integer
        Number of threads to use for PCA. Default is None for
        single-threaded computation.
        
    Returns: 
    ---------------------- 
    snr: float
        Computed SNR.
    noise_factor: float
        Computed noise factor.
    """

    # Get shape and origin
    cube_shape = np.shape(image_cube)
    origin = [cube_shape[2]/2 - 0.5, cube_shape[1]/2 - 0.5]
    
    '''
    IWA_mask =  hf.circular_mask(origin, aperture_radius, cube_shape[1], cube_shape[2])
    med_image_unsubtracted = np.nanmedian(image_cube, axis=0)
    stellar_flux = np.nansum(med_image_unsubtracted[IWA_mask])
    stellar_peak_flux = np.nanmax(med_image_unsubtracted[IWA_mask])
    '''

    image_cube[np.isnan(image_cube)] = 0
    
    # Do PCA on the relevant annulus
    if delta_rot == 0 and n_segments <= 1:
    
        med_image = pca_annulus(image_cube, para_angles, ncomp=ncomp, r_guess=source_radius,
                                annulus_width=1.05*2*aperture_radius,
                                nproc=nproc, svd_mode='eigen', imlib='opencv')
    
    else:
        
        med_image = pca_annular(image_cube, para_angles, fwhm=aperture_radius*2, ncomp=ncomp,
                                asize=1.05*2*aperture_radius, verbose=False, delta_rot=delta_rot,
                                radius_int=source_radius-aperture_radius, n_segments=n_segments,
                                nproc=nproc, svd_mode='eigen', imlib='opencv')

    # Compute angles at which to create apertures
    circle_pos = np.linspace(0, 2*np.pi,
                             int(np.floor(np.pi*source_radius/aperture_radius))+1)[:-1]
    fluxes = np.zeros(len(circle_pos))

    # Measure flux in each aperture by creating a mask
    for i in range(len(circle_pos)):
    
        mask = hf.circular_mask((origin[0] + source_radius*np.cos(circle_pos[i] +
                                                                  source_angle*np.pi/180),
                              origin[1] + source_radius*np.sin(circle_pos[i] +
                                                                  source_angle*np.pi/180)),
                             aperture_radius, np.shape(med_image)[0], np.shape(med_image)[1])
        if i == 0:

            source_peak_flux = (np.nanmax(med_image[mask]))
            
        fluxes[i] = np.nansum(med_image[mask])

    # Compute SNR
    noise_factor = (np.std(fluxes[1:])*np.sqrt(1 + 1/(len(fluxes) - 1)))

    snr = (fluxes[0] - np.mean(fluxes[1:]))/noise_factor

    return snr, noise_factor

#------------------------

def contrast_curve(frames, para_angles, reffits, array_shape, curve_prior,
                   injection_radii, injection_angles, aperture_radius,
                   ncomp=5, delta_rot=0, n_segments=1, highpassrad=None,
                   tolerance=0.05, high_buffer=2, max_iterations=10, 
                   add_info=[None, None], nproc=1, threadcount=6):
    """
    Computes a contrast curve for an image sequence.
    
    Parameters (contained inside a tuple):
    ----------------------
    frames: List of file paths or 3D image array
    para_angles: list or array
        List of parallactic angles corresponding to each file
        in frames
    reffits: 1D numpy array
        Array of airy disk fitting parameters corresponding to
        each file in frames
    array_shape: integer tuple
        Tuple containing image dimensions, from numpy.shape
    curve_prior: 1D numpy array
        A contrast curve used as a starting assumption for performing
        SNR measurements.
    injection_radii: 1D numpy array
        Radii from the stellar psf in which to inject sources.
    injection_angles: 1D numpy array
        Position angles at which to inject sources.
    aperture_radius: integer
        Radius of SNR measurement apertures in pixels.
    ncomp (optional): integer
        Number of KLIP components.
    delta_rot (optional): float
        Excludes images with angles within delta_rot from
        the PSF library for each image. Default is 0, which allows
        for the PSF library to be only computed once.
    n_segments (optional): integer
        Number of segments in each annulus for PCA.
    highpassrad (optional): integer
        Highpass filter radius. None by default, disabling highpass
        filtering.
    tolerance (optional): float
        Allowable tolerance for computing the SNR as a percent.
    high_buffer (optional): float
        Multiplicative factor applied to curve_prior to ensure that
        modelling starts with a high SNR source.
    max_iterations (optional): integer
        Maximum number of iterations allowed when recursively
        calculating source SNR.
    add_info (optional): list
        Additional information to include in error files.
    nproc (optional): integer
        Number of processors to employ in vip_hci pca implementations.
    threadcount (optional): integer
        Number of threads to employ in multithreading.
        Default value is 6 threads.        
        
    Returns: 
    ---------------------- 
    approx_contrasts:
    len(injection_radii) x len(injection_angles) 2D numpy array
        Measured contrasts at each injection radius and angle
    snrs:
    len(injection_radii) x len(injection_angles) 2D numpy array
        Associated SNRs with each contrast in approx_contrasts.
    """
    
    # Stack angles and injection radii
    coordlist = np.vstack(np.asarray(np.meshgrid(injection_radii, injection_angles)).T)

    if threadcount > 1:
        with Pool(threadcount) as pool:
            (approx_contrast_list,
             snrs_list) = zip(*tqdm(pool.imap(ForwardModel((frames, para_angles, reffits,
                                                            array_shape, curve_prior,
                                                            aperture_radius, ncomp, delta_rot,
                                                            n_segments, highpassrad,
                                                            tolerance, high_buffer, max_iterations,
                                                            add_info, 1)), coordlist),
                                    total=len(coordlist)))
    else:
        (snrs_list,
         approx_contrast_list) = (np.zeros((len(injection_radii)*len(injection_angles))),
                                  np.zeros((len(injection_radii)*len(injection_angles))))

        contrast_gen = ForwardModel((frames, para_angles, reffits, array_shape, curve_prior,
                                     aperture_radius, ncomp, delta_rot, n_segments, highpassrad,
                                     tolerance, high_buffer, max_iterations, add_info, nproc))
    
        for i in tqdm(range(len(coordlist))):
            
            approx_contrast_list[i], snrs_list[i] = contrast_gen(coordlist[i])

    approx_contrasts = np.asarray(approx_contrast_list).reshape(len(injection_radii),
                                                                len(injection_angles))
    snrs = np.asarray(snrs_list).reshape(len(injection_radii), len(injection_angles))

    return approx_contrasts, snrs

def snr_map(image, aperture_radius=14, threadcount=50):

    array_shape = np.shape(image)
    snrmap = np.zeros(array_shape)

    wx = np.linspace(0, array_shape[0]-1, array_shape[0])
    wy = np.linspace(0, array_shape[1]-1, array_shape[1])
    wx, wy = np.meshgrid(wx, wy)
    
    tup_arr = np.array([wx, wy])
    tup_arr = tup_arr.reshape(2, np.shape(tup_arr)[1]*np.shape(tup_arr)[2]).T

    origin = [array_shape[1]/2 - 0.5, array_shape[0]/2 - 0.5]

    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
            snrs, noise_factors, = \
                zip(*tqdm(pool.imap(FluxSNR((image, array_shape,
                                        origin, aperture_radius)),
                               tup_arr)))

    return np.asarray(snrs), np.asarray(noise_factors)
