import os
import numpy as np
from astropy.io import fits
from pyklip.klip import klip_math, collapse_data, estimate_movement, define_annuli_bounds
from pyklip.parallelized import rotate_imgs, generate_noise_maps
#from multiprocessing.pool import ThreadPool as Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
import scipy.ndimage as ndi
from tqdm.auto import tqdm

def klip_dataset(filepaths, PAs,  origin, framew=512, frameh=512, OWA=None, IWA=30,\
                 mode='ADI', outputdir=".", fileprefix="", annuli=1, subsections=1, movement=1,
                 numbasis=None, numthreads=None, minrot=0, maxrot=360,annuli_spacing="constant", maxnumbasis=None, corr_smooth=1, algo='klip', skip_derot=False, flipx=False, time_collapse="mean", compute_noise_cube=False,verbose=True):

    # Checking parameters
    data_input = np.zeros((len(filepaths), framew, frameh))
    for i in range(len(filepaths)):
        hdul = fits.open(filepaths[i])
        frame = hdul[0].data
        data_input[i] = frame
        hdul.close()

    filenums = np.linspace(0, len(filepaths)-1, len(filepaths))
    
    # empca currently does not support movement or minrot
    if algo.lower() == 'empca' and (minrot != 0 or movement != 0):
        raise ValueError('empca currently does not support movement, minrot selection criteria, '
                         'must be set to 0')
    elif algo.lower() == 'none':
        # remove some psfsubtraction params
        movement = 0
        minmove = 0
        numbasis = [1]

    # check which algo we will use and whether the inputs are correct
    if algo.lower() == 'klip':
        pass
    elif algo.lower() == 'empca':
        pass
    elif algo.lower() == 'nmf':
        # check to see the correct nmf packages are installed 
        import pyklip.nmf_imaging as nmf_imaging
    elif algo.lower() == 'nmf_jax':
        # check to see the correct nmf_jax packages are installed 
        try:
            import pyklip.nmf_imaging_JAX as nmf_imaging_jax
        except:
            raise ImportError("Failed to import NMF JAX. Please ensure that JAX is properly installed.")
    elif algo.lower() == 'none':
        pass
    else:
        raise ValueError("Algo {0} is not supported".format(algo))

    if numbasis is None:
        totalimgs = data_input.shape[0]
        maxbasis = np.min([totalimgs, 100]) # only going up to 100 KL modes by default
        numbasis = np.arange(1, maxbasis + 5, 10)
        print("KL basis not specified. Using default.", numbasis)
    else:
        if hasattr(numbasis, "__len__"):
            numbasis = np.array(numbasis)
        else:
            numbasis = np.array([numbasis])

    time_collapse = time_collapse.lower()
    weighted = "weighted" in time_collapse # boolean whether to use weights

    if corr_smooth < 0:
        raise ValueError("corr_smooth needs be non-negative. Supplied value is {0}".format(corr_smooth))

    # if no outputdir specified, then current working directory (don't want to write to '/'!)
    if outputdir == "":
        outputdir = "."

    # save klip parameters as a string

    maxbasis_str = maxnumbasis if maxnumbasis is not None else np.max(numbasis) # prefer to use maxnumbasis if possible

    #save all bad pixels
    allnans = np.where(np.isnan(data_input))

    dims = data_input.shape
    x, y = np.meshgrid(np.arange(dims[2] * 1.0), np.arange(dims[1] * 1.0))
    nanpix = np.where(np.isnan(data_input[0]))

    if OWA is None:
        full_image = True # reduce the full image
        # define OWA as either the closest NaN pixel or edge of image if no NaNs exist
        if np.size(nanpix) == 0:
            OWA = np.sqrt(np.max((x - origin[0]) ** 2 + (y - origin[1]) ** 2))
        else:
            # grab the NaN from the 1st percentile (this way we drop outliers)
            OWA = np.sqrt(np.percentile((x[nanpix] - origin[0]) ** 2 + (y[nanpix] - origin[1]) ** 2, 1))
    else:
        full_image = False # don't reduce the full image, only up the the IWA

    #calculate the annuli ranges
    rad_bounds = define_annuli_bounds(annuli, IWA, OWA, annuli_spacing)

    # if OWA wasn't passed in, we're going to assume we reduce the full image, so last sector emcompasses everything
    if full_image:
        # last annulus should mostly emcompass everything
        rad_bounds[annuli - 1] = (rad_bounds[annuli - 1][0], data_input[0].shape[0])

    #divide annuli into subsections
    dphi = 2 * np.pi / subsections
    phi_bounds = [[dphi * phi_i - np.pi, dphi * (phi_i + 1) - np.pi] for phi_i in range(subsections)]
    phi_bounds[-1][1] = np.pi


# HERE's WHERE MEMORY PROBLEMS START

    #create a coordinate system. Can use same one for all the images because they have been aligned and scaled
    x, y = np.meshgrid(np.arange(dims[2] * 1.0), np.arange(dims[1] * 1.0))
    x.shape = (x.shape[0] * x.shape[1]) #Flatten
    y.shape = (y.shape[0] * y.shape[1])
    
    r = np.sqrt((x - origin[0]) ** 2 + (y - origin[1]) ** 2)
    phi = np.arctan2(y - origin[1], x - origin[0])
    phi = (phi % (2 * np.pi)) - np.pi

    #export some of klip.klip_math functions to here to minimize computation repeats

    #load aligned images for this wavelength

    data_input.shape = (dims[0], dims[1] * dims[2])
    
    #list to store each threadpool task
    klipped_values = []
    klipped_indices = []
    klipped_imgs = np.nan*np.ones((dims[1] * dims[2], dims[0]) + numbasis.shape)

    #as each is finishing, queue up the aligned data to be processed with KLIP

    if verbose is True:
        print("Queuing for KLIP")

    for radstart, radend in rad_bounds:
        for phistart, phiend in phi_bounds:
                klipped_value, klipped_index = _klip_section_multifile(data_input, dims, PAs, r, phi, filenums, numbasis,
                                                                    maxnumbasis,
                                                                    radstart, radend, phistart, phiend, movement,
                                                                    origin, minrot, maxrot,
                                                                    mode, corr_smooth, algo, numthreads, verbose)
                klipped_values.append(klipped_value)
                klipped_indices.append(klipped_index)
            
    for i in range(len(klipped_indices)):
        klipped_imgs[klipped_indices[i]] = np.swapaxes(klipped_values[i], 0, 1)
    klipped_values, klipped_indices = None, None
    
    klipped_imgs = np.swapaxes(klipped_imgs, 0, 1)
    klipped_imgs.shape = dims + numbasis.shape

    #close to pool now and make sure there's no processes still running (there shouldn't be or else that would be bad)
    if verbose is True:
        print("Closing threadpool")

    #finished. Let's reshape the output images
    #move number of KLIP modes as leading axis (i.e. move from shape (N,y,x,b) to (b,N,y,x)
    klipped_imgs = np.rollaxis(klipped_imgs.reshape((dims[0], dims[1], dims[2], numbasis.shape[0])), 3)

    #restore bad pixels
    klipped_imgs[:, allnans[0], allnans[1], allnans[2]] = np.nan

    # calculate weights for weighted mean if necessary
    if compute_noise_cube:
        print("Computing weights for weighted collapse")
        # figure out ~how wide to make it
        annuli_widths = [annuli_bound[1] - annuli_bound[0] for annuli_bound in rad_bounds]
        dr_spacing = np.min(annuli_widths)
        # generate all teh noise maps. We need to collapse the klipped_imgs into 3-D to easily do this
        klipped_imgs_shape = klipped_imgs.shape
        klipped_imgs_flatten = klipped_imgs.reshape([klipped_imgs_shape[0]*klipped_imgs_shape[1], klipped_imgs_shape[2], klipped_imgs_shape[3]])
        noise_frames = generate_noise_maps(klipped_imgs_flatten, origin, dr_spacing, IWA=IWA, OWA=rad_bounds[-1][1], numthreads=numthreads)
        # reform the 4-D cubes
        noise_frames = noise_frames.reshape(klipped_imgs_shape) # reshape into a cube with same shape as klipped_imgs
    else:
        noise_frames = np.array([1.])

    # Clear memory
    klipped_imgs = np.asarray(klipped_imgs)
        
    stddev_frames = noise_frames

    # convert lists to numpy arrays
    stddev_frames = np.array(stddev_frames)

    # TODO: handling of only a single numbasis
    # derotate all the images
    # flatten so it's just a 3D array (collapse KL and Nframes dimensions)
    oldshape = klipped_imgs.shape

    klipped_imgs = klipped_imgs.reshape(oldshape[0]*oldshape[1], oldshape[2], oldshape[3])
    if weighted:
        # do the same for the stddev frames
        stddev_frames = stddev_frames.reshape(oldshape[0]*oldshape[1], oldshape[2], oldshape[3])

    # we need to duplicate PAs and centers for the different KL mode cutoffs we supplied
    flattend_parangs = np.tile(PAs, oldshape[0])
    
    flattened_centers = np.tile(np.asarray([(origin[0], origin[1])]*len(filepaths)).reshape(oldshape[1]*2), oldshape[0]).reshape(oldshape[1]*oldshape[0],2)

    # if skipping derotating, set all rotations to 0
    if skip_derot:
        flattend_parangs[:] = 0

    # parallelized rotate images
    print("Derotating Images...")
    klipped_imgs = rotate_imgs(klipped_imgs, flattend_parangs, flattened_centers, numthreads=numthreads, flipx=flipx, new_center=origin)
    # re-expand the images in num cubes/num wvs (num KLmode cutoffs, num cubes, num wvs, y, x)
    klipped_imgs = klipped_imgs.reshape(oldshape[0], oldshape[1], oldshape[2], oldshape[3])

    # rotate the weights too if necessary
    if weighted:
        stddev_frames = rotate_imgs(stddev_frames, flattend_parangs, flattened_centers, numthreads=numthreads, flipx=flipx, new_center=origin)
        stddev_frames = stddev_frames.reshape(oldshape[0], oldshape[1], oldshape[2], oldshape[3])

    # valid output path and write iamges
    outputdirpath = os.path.realpath(outputdir)
    print("Writing Images to directory {0}".format(outputdirpath))

    # create weights for each pixel. If we aren't doing weighted mean, weights are just ones
    pixel_weights = 1./stddev_frames**2

    KLmode_cube = collapse_data(klipped_imgs, pixel_weights, axis=1, collapse_method=time_collapse)

    filename = '{}-KLmodes-all.fits'.format(fileprefix)
    filepath = os.path.join(outputdirpath, filename)

    newhdul = fits.HDUList([fits.PrimaryHDU(data=KLmode_cube)])
    newhdul.writeto(filepath, overwrite=True)
    newhdul.close() 
    
    return


def _klip_section_multifile(aligned_imgs, dims, parangs, r, phi, filenums, numbasis, maxnumbasis, radstart, radend, phistart,
                            phiend, minmove, ref_center, minrot, maxrot, mode, corr_smooth=1, algo='klip', numthreads=12, verbose=True):
    """
    Runs klip on a section of the image for all the images of a given wavelength.
    Bigger size of atomization of work than _klip_section but saves computation time and memory. Currently no need to
    break it down even smaller when running on machines on the order of 32 cores.

    Args:
        numbasis: number of KL basis vectors to use (can be a scalar or list like). Length of b
        maxnumbasis: if not None, maximum number of KL basis/correlated PSFs to use for KLIP. Otherwise, use max(numbasis)    
        radstart: inner radius of the annulus (in pixels)
        radend: outer radius of the annulus (in pixels)
        phistart: lower bound in CCW angle from x axis for the start of the section
        phiend: upper boundin CCW angle from y axis for the end of the section
        minmove: minimum movement between science image and PSF reference image to use PSF reference image (in pixels)
        ref_center: 2 element list for the center of the science frames. Science frames should all be aligned.
        minrot: minimum PA rotation (in degrees) to be considered for use as a reference PSF (good for disks)
        maxrot: maximum PA rotation (in degrees) to be considered for use as a reference PSF (temporal variability)
        mode: one of ['ADI', 'SDI', 'ADI+SDI'] for ADI, SDI, or ADI+SDI
        corr_smooth (float): size of sigma of Gaussian smoothing kernel (in pixels) when computing most correlated PSFs. If 0, no smoothing
        algo (str): algorithm to use ('klip', 'nmf', 'empca','nmf_jax')
        verbose (bool): if True, prints out warnings

    Returns:
        returns True on success, False on failure. Does not return whether KLIP on each individual image was sucessful.
        Saves data to output array as defined in _tpool_init()
    """
    #grab the pixel location of the section we are going to anaylze
    section_ind = np.where((r >= radstart) & (r < radend) & (phi >= phistart) & (phi < phiend))
    if np.size(section_ind) <= 1:
        if verbose is True:
            print("section is too small ({0} pixels), skipping...".format(np.size(section_ind)))
        return False

    '''
    if algo.lower() == 'empca':

        #TODO: include scidata_indices selection in here
        try:
            full_model = np.zeros(ref_psfs.shape)
            ref_psfs[np.isnan(ref_psfs)] = 0.
            good_ind = np.sum(ref_psfs > 0., axis=0) > numbasis[-1]

            # set weights for empca
            # TODO: change the inner and outer suppression angle to not be hard coded
            rflat = np.reshape(r[section_ind[0]], -1) # 1D array of radii, size=number of pixels in the section
            weights = empca.set_pixel_weights(ref_psfs[:, good_ind], rflat[good_ind], mode='standard', inner_sup=17,
                                              outer_sup=66)

            # run empca reduction
            output_imgs_np = _arraytonumpy(output, (output_shape[0], output_shape[1] * output_shape[2], output_shape[3]), dtype=dtype)
            for i, rank in enumerate(numbasis):
                # get indices of the image section that have enough finite values along the time dimension
                good_ind_model = empca.weighted_empca(ref_psfs[:, good_ind], weights=weights, niter=15, nvec=rank)
                full_model[:, good_ind] = good_ind_model
                output_imgs_np[:, section_ind[0], i] = aligned_imgs[:, section_ind[0]] - full_model

        except (ValueError, RuntimeError, TypeError) as err:
            print(err.args)
            return False

        return True
    '''
    
    #do the same for the reference PSFs
    #playing some tricks to vectorize the subtraction of the mean for each row
    if algo.lower() == 'nmf' or algo.lower() == 'nmf_jax': # do not do mean subtraction for NMF
        ref_psfs_mean_sub = aligned_imgs[:,  section_ind[0]]
    else:
        ref_psfs_mean_sub = aligned_imgs[:,  section_ind[0]] - np.nanmean(aligned_imgs[:,  section_ind[0]], axis=1)[:, None]
        
    ref_psfs_mean_sub[np.where(np.isnan(ref_psfs_mean_sub))] = 0

    #calculate the covariance matrix for the reference PSFs
    #note that numpy.cov normalizes by p-1 to get the NxN covariance matrix
    #we have to correct for that in the klip.klip_math routine when consturcting the KL
    #vectors since that's not part of the equation in the KLIP paper
    covar_psfs = np.cov(ref_psfs_mean_sub)
        
    if ref_psfs_mean_sub.shape[0] == 1:
        # EDGE CASE: if there's only 1 image, we need to reshape to covariance matrix into a 2D matrix
        covar_psfs = covar_psfs.reshape((1,1))

    if corr_smooth > 0:
        # calcualte the correlation matrix, with possible smoothing  
        aligned_imgs_3d = aligned_imgs.reshape([aligned_imgs.shape[0], dims[-2], dims[-1]]) # make a cube that's not flattened in spatial dimension
        # smooth only the square that encompasses the segment
        # we need to figure where that is
        # figure out the smallest square that encompasses this sector
        blank_img = np.ones(dims[-2:]) * np.nan
        blank_img.ravel()[section_ind] = 0
        y_good, x_good = np.where(~np.isnan(blank_img))
        ymin = np.min(y_good)
        ymax = np.max(y_good)
        xmin = np.min(x_good)
        xmax = np.max(x_good)
        blank_img_crop = blank_img[ymin:ymax+1, xmin:xmax+1]
        section_ind_smooth_crop = np.where(~np.isnan(blank_img_crop))
        # now that we figured out only the region of interest for each image to smooth, let's smooth that region'
        ref_psfs_smoothed = []
        for aligned_img_2d in aligned_imgs_3d:
            smooth_sigma = 1
            smoothed_square_crop = ndi.gaussian_filter(aligned_img_2d[ymin:ymax+1, xmin:xmax+1], smooth_sigma)
            smoothed_section = smoothed_square_crop[section_ind_smooth_crop]
            smoothed_section[np.isnan(smoothed_section)] = 0
            ref_psfs_smoothed.append(smoothed_section)

        # Clear memory
        aligned_imgs_3d = None
    
        corr_psfs = np.corrcoef(ref_psfs_smoothed)
        if ref_psfs_mean_sub.shape[0] == 1:
            # EDGE CASE: if there's only 1 image, we need to reshape the correlation matrix into a 2D matrix
            corr_psfs = corr_psfs.reshape((1,1))
            
        # smoothing could have caused some ref images to have all 0s
        # which would give them correlation matrix entries of NaN
        # 0 them out for now.
        corr_psfs[np.where(np.isnan(corr_psfs))] = 0
    else:
        # if we don't smooth, we can use the covariance matrix to calculate the correlation matrix. It'll be slightly faster
        covar_diag = np.diagflat(1./np.sqrt(np.diag(covar_psfs)))
        corr_psfs = np.dot( np.dot(covar_diag, covar_psfs ), covar_diag)
        
    output_klipped = np.zeros((len(filenums),len(section_ind[0])) + numbasis.shape)

    iterator = zip(range(len(filenums)), parangs, filenums)
    '''
    with Pool(numthreads) as pool:
        
        results,_ = zip(*tqdm(pool.imap(perFileKLIP((aligned_imgs, ref_psfs_mean_sub,parangs, filenums, section_ind, covar_psfs, corr_psfs, (radstart + radend) / 2.0, numbasis, maxnumbasis, minmove, minrot, maxrot, mode, algo, verbose)), iterator), total=len(filenums)))

    for i in range(len(results)):
        output_klipped[i] = results[i]
    results = []

    '''
    # Set up ThreadPoolExecutor
    n_workers = min(numthreads, len(filenums))
    
    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as exe:
        # Run evaluate params 
        futures = [exe.submit(_klip_section_multifile_perfile, aligned_imgs, ref_psfs_mean_sub, parangs, filenums, file_index, section_ind, covar_psfs, corr_psfs,
                                            parang, filenum, (radstart + radend) / 2.0, numbasis,
                                            maxnumbasis, minmove, minrot, maxrot, mode,
                                            algo, verbose)
                   for file_index, parang, filenum in zip(range(len(filenums)), parangs, filenums)]

        results = []
        with tqdm(total=len(futures), desc="Performing klip subtraction", leave=True) as pbar:
            for f in as_completed(futures):
                results.append(f.result())
                pbar.update(1)
    results = sorted(results, key=lambda x: x[0])
    for i in range(len(results)):
        output_klipped[i] = results[i][-1]
    results = []


    '''
    # Non-parallelized
    for file_index, parang, filenum in tqdm(zip(range(len(filenums)), parangs, filenums), total=len(filenums)):
        try:
            output_klipped[file_index]  =  _klip_section_multifile_perfile(aligned_imgs,ref_psfs_mean_sub,  parangs, filenums, file_index, section_ind, covar_psfs, corr_psfs,
                                            parang, filenum, (radstart + radend) / 2.0, numbasis,
                                            maxnumbasis, minmove, minrot, maxrot, mode,
                                            algo, verbose)[-1]
        except (ValueError, RuntimeError, TypeError) as err:
            print(err.args)
            return False
    '''
    return output_klipped, section_ind

class perFileKLIP(object):
    
    def __init__(self, vars):
        
        self.imgs = vars[0]
        self.ref_psfs = vars[1]
        self.vars = vars[2:]
    
    def __call__(self, params):

        pa_imgs, filenums_imgs, section_ind, covar,  corr, avg_rad,\
                                    numbasis, maxnumbasis, minmove, minrot, maxrot, mode, algo, verbos = self.vars

        img_num, parang, filenum = params
            
        # grab the files suitable for reference PSF
        # load shared arrays for wavelengths, PAs, and filenumbers

        # calculate average movement in this section for each PSF reference image w.r.t the science image
        moves = estimate_movement(avg_rad, parang, pa_imgs, mode=mode)
        # check all the PSF selection criterion
        # enough movement of the astrophyiscal source
        goodmv = (moves >= minmove)
    
        # enough field rotation
        if minrot > 0:
            goodmv = (goodmv) & (np.abs(pa_imgs - parang) >= minrot)
    
        good_file_ind = np.where(goodmv)
        # Remove reference psfs if they are mostly nans
        ref2rm = np.where(np.nansum(np.isfinite(self.ref_psfs[good_file_ind[0], :]),axis=1) < 5)[0]
        good_file_ind = (np.delete(good_file_ind[0],ref2rm),)
        if (np.size(good_file_ind[0]) < 1):
            if verbose is True:
                print("less than 1 reference PSFs available for minmove={0}, skipping...".format(minmove))
            return False
        # pick out a subarray. Have to play around with indicies to get the right shape to index the matrix
        covar_files = covar[good_file_ind[0].reshape(np.size(good_file_ind), 1), good_file_ind[0]]
    
        # pick only the most correlated reference PSFs if there's more than enough PSFs
        if maxnumbasis is None:
            maxnumbasis = np.max(numbasis)
        maxbasis_possible = np.size(good_file_ind)
    
        # do we want to downselect out of all the possible references
        if maxbasis_possible > maxnumbasis:
            # grab the x-correlation with the sci img for valid PSFs
            xcorr = corr[img_num, good_file_ind[0]]
    
            sort_ind = np.argsort(xcorr)
            closest_matched = sort_ind[-maxnumbasis:]  # sorted smallest first so need to grab from the end
    
            # grab smaller set of reference PSFs
            ref_psfs_selected = self.ref_psfs[good_file_ind[0][closest_matched], :]
            # grab the new and smaller covariance matrix
            covar_files = covar_files[closest_matched.reshape(np.size(closest_matched), 1), closest_matched]
    
        else:
            # else just grab the reference PSFs for all the valid files
            ref_psfs_selected = ref_psfs[good_file_ind[0], :]
    
        # output_images has shape (N, y*x, b) and not (N, y, x, b) as normal
        
        # run KLIP
        try:
            if algo.lower() == 'klip':
                klipped = klip_math(self.imgs[img_num, section_ind[0]], ref_psfs_selected, numbasis, covar_psfs=covar_files)
            elif algo.lower() == 'nmf':
                import pyklip.nmf_imaging as nmf_imaging
                klipped = nmf_imaging.nmf_math(self.imgs[img_num, section_ind].ravel(), ref_psfs_selected, componentNum=numbasis)
            elif algo.lower() == 'nmf_jax':
                import pyklip.nmf_imaging_JAX as nmf_imaging_jax
                klipped = nmf_imaging_jax.nmf_func(self.imgs[img_num, section_ind].ravel(), ref_psfs_selected, componentNum=numbasis)
            elif algo.lower() == "none":
                klipped = np.array([self.imgs[img_num, section_ind[0]] for _ in range(len(numbasis))]) # duplicate by requested numbasis
                klipped = klipped.T # retrun in shape (p, b) as expected
        except (ValueError, RuntimeError, TypeError) as err:
            print(err.args)
            return False, img_num
    
        # write to output 
        #output_imgs[img_num, section_ind[0], :] = klipped
    
        return klipped, img_num#True



def _klip_section_multifile_perfile(aligned_imgs, ref_psfs, pa_imgs, filenums_imgs, img_num, section_ind, covar,  corr, parang, filenum, avg_rad,
                                    numbasis, maxnumbasis, minmove, minrot, maxrot, mode, algo, verbose):
    """
    Imitates the rest of _klip_section for the multifile code. Does the rest of the PSF reference selection and runs KLIP.

    Args:
        img_num: file index for the science image to process
        section_ind: np.where(pixels are in this section of the image). Note: coordinate system is collapsed into 1D
        ref_psfs: reference psf images of this section
        covar: the covariance matrix of the reference PSFs. Shape of (N,N)
        corr: the correlation matrix of the refernece PSFs. Shape of (N,N)
        parang: PA of science iamage
        filenum (int): file number of science image
        avg_rad: average radius of this annulus
        numbasis: number of KL basis vectors to use (can be a scalar or list like). Length of b
        maxnumbasis: if not None, maximum number of KL basis/correlated PSFs to use for KLIP. Otherwise, use max(numbasis)           
        minmove: minimum movement between science image and PSF reference image to use PSF reference image (in pixels)
        mode: one of ['ADI', 'SDI', 'ADI+SDI'] for ADI, SDI, or ADI+SDI
        verbose (bool): if True, prints out error messages

    Returns:
        return True on success, False on failure.
        Saves image to output array defined in _tpool_init()
    """
        
    # grab the files suitable for reference PSF
    # load shared arrays for wavelengths, PAs, and filenumbers
    # calculate average movement in this section for each PSF reference image w.r.t the science image
    moves = estimate_movement(avg_rad, parang, pa_imgs, mode=mode)
    # check all the PSF selection criterion
    # enough movement of the astrophyiscal source
    goodmv = (moves >= minmove)

    # enough field rotation
    if minrot > 0:
        goodmv = (goodmv) & (np.abs(pa_imgs - parang) >= minrot)

    good_file_ind = np.where(goodmv)
    # Remove reference psfs if they are mostly nans
    ref2rm = np.where(np.nansum(np.isfinite(ref_psfs[good_file_ind[0], :]),axis=1) < 5)[0]
    good_file_ind = (np.delete(good_file_ind[0],ref2rm),)
    if (np.size(good_file_ind[0]) < 1):
        if verbose is True:
            print("less than 1 reference PSFs available for minmove={0}, skipping...".format(minmove))
        return False
    # pick out a subarray. Have to play around with indicies to get the right shape to index the matrix
    covar_files = covar[good_file_ind[0].reshape(np.size(good_file_ind), 1), good_file_ind[0]]

    # pick only the most correlated reference PSFs if there's more than enough PSFs
    if maxnumbasis is None:
        maxnumbasis = np.max(numbasis)
    maxbasis_possible = np.size(good_file_ind)

    # load input/output data

    # do we want to downselect out of all the possible references
    if maxbasis_possible > maxnumbasis:
        # grab the x-correlation with the sci img for valid PSFs
        xcorr = corr[img_num, good_file_ind[0]]

        sort_ind = np.argsort(xcorr)
        closest_matched = sort_ind[-maxnumbasis:]  # sorted smallest first so need to grab from the end

        # grab smaller set of reference PSFs
        ref_psfs_selected = ref_psfs[good_file_ind[0][closest_matched], :]
        # grab the new and smaller covariance matrix
        covar_files = covar_files[closest_matched.reshape(np.size(closest_matched), 1), closest_matched]

    else:
        # else just grab the reference PSFs for all the valid files
        ref_psfs_selected = ref_psfs[good_file_ind[0], :]

    # output_images has shape (N, y*x, b) and not (N, y, x, b) as normal

    # run KLIP
    try:
        if algo.lower() == 'klip':
            klipped = klip_math(aligned_imgs[img_num, section_ind[0]], ref_psfs_selected, numbasis, covar_psfs=covar_files)
        elif algo.lower() == 'nmf':
            import pyklip.nmf_imaging as nmf_imaging
            klipped = nmf_imaging.nmf_math(aligned_imgs[img_num, section_ind].ravel(), ref_psfs_selected, componentNum=numbasis)
        elif algo.lower() == 'nmf_jax':
            import pyklip.nmf_imaging_JAX as nmf_imaging_jax
            klipped = nmf_imaging_jax.nmf_func(aligned_imgs[img_num, section_ind].ravel(), ref_psfs_selected, componentNum=numbasis)
        elif algo.lower() == "none":
            klipped = np.array([aligned_imgs[img_num, section_ind[0]] for _ in range(len(numbasis))]) # duplicate by requested numbasis
            klipped = klipped.T # retrun in shape (p, b) as expected
    except (ValueError, RuntimeError, TypeError) as err:
        print(err.args)
        return False

    return [img_num, klipped]
