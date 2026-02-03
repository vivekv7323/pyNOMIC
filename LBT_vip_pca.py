import numpy as np
from multiprocessing import cpu_count
from typing import Tuple, List, Union
from enum import Enum
from dataclasses import dataclass
from vip_hci.psfsub.svd import get_eigenvectors, svd_wrapper

from vip_hci.psfsub import PCA_ANNULAR_Params, PCA_Params
from vip_hci.config import time_ini, timing, check_array, check_enough_memory
from vip_hci.config.utils_conf import pool_map, iterable
from vip_hci.config.utils_param import setup_parameters, separate_kwargs_dict
from vip_hci.preproc import (cube_derotate, check_pa_vector,
                       check_scal_vector)
from pyklip.parallelized import rotate_imgs
#from pyklip.klip import rotate as klip_rotate
from vip_hci.preproc.derotation import _find_indices_adi, _define_annuli
from vip_hci.stats import descriptive_stats
from vip_hci.var import get_annulus_segments, matrix_scaling, prepare_matrix, reshape_matrix
from vip_hci.config.paramenum import SvdMode, Imlib, Interpolation, Collapse, ALGO_KEY
import matplotlib.pyplot as plt

AUTO = "auto"
'''
def rotate_imgs(imgs, angles, centers, new_center=None, numthreads=None, flipx=False, hdrs=None):
    """
    derotate a sequences of images by their respective angles

    Args:
        imgs: array of shape (N,y,x) containing N images
        angles: array of length N with the angle to rotate each frame. Each angle should be CCW in degrees.
        centers: array of shape N,2 with the [x,y] center of each frame
        new_centers: a 2-element array with the new center to register each frame. Default is middle of image
        numthreads: number of threads to be used
        flipx: flip the x axis after rotation if desired
        hdrs: array of N wcs astrometry headers

    Returns:
        derotated: array of shape (N,y,x) containing the derotated images
    """

    # klip.rotate(img, -angle, oldcenter, [152,152]) for img, angle, oldcenter
    # multithreading the rotation for each image
    derotated = np.array([klip_rotate(img, angle, center, new_center, flipx, None)
                 for img, angle, center in zip(imgs, angles, centers)])

    return derotated
'''

def pca_annular(*all_args: List, **all_kwargs: dict):
    """PCA model PSF subtraction for ADI, ADI+RDI or ADI+mSDI (IFS) data.

    The PCA model is computed locally in each annulus (or annular sectors
    according to ``n_segments``). For each sector we discard reference frames
    taking into account a parallactic angle threshold (``delta_rot``) and
    optionally a radial movement threshold (``delta_sep``) for 4d cubes.

    For ADI+RDI data, it computes the principal components from the reference
    library/cube, forcing pixel-wise temporal standardization. The number of
    principal components can be automatically adjusted by the algorithm by
    minimizing the residuals inside each patch/region.

    References: [AMA12]_ for PCA-ADI; [ABS13]_ for PCA-ADI in concentric annuli
    considering a parallactic angle threshold; [CHR19]_ for PCA-ASDI and
    PCA-SADI in one or two steps.

    Parameters
    ----------
    all_args: list, optional
        Positionnal arguments for the PCA annular algorithm. Full list of
        parameters below.
    all_kwargs: dictionary, optional
        Mix of keyword arguments that can initialize a PCA_ANNULAR_Params and
        the optional 'rot_options' dictionary, with keyword values for
        "border_mode", "mask_val", "edge_blend", "interp_zeros", "ker" (see
        documentation of ``vip_hci.preproc.frame_rotate``). Can also contain a
        PCA_ANNULAR_Params named as `algo_params`.

    PCA annular parameters
    ----------
    cube : numpy ndarray, 3d or 4d
        Input cube.
    angle_list : 1d numpy ndarray
        Vector of derotation angles to align North up in your cube images.
    cube_ref : numpy ndarray, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    scale_list : numpy ndarray, 1d, optional
        If provided, triggers mSDI reduction. These should be the scaling
        factors used to re-scale the spectral channels and align the speckles
        in case of IFS data (ADI+mSDI cube). Usually, these can be approximated
        by the last channel wavelength divided by the other wavelengths in the
        cube (more thorough approaches can be used to get the scaling factors,
        e.g. with ``vip_hci.preproc.find_scal_vector``).
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the
        central circular region is discarded.
    fwhm : float, optional
        Size of the FWHM in pixels. Default is 4.
    asize : float, optional
        The size of the annuli, in pixels.
    n_segments : int or list of ints or 'auto', optional
        The number of segments for each annulus. When a single integer is given
        it is used for all annuli. When set to 'auto', the number of segments is
        automatically determined for every annulus, based on the annulus width.
    delta_rot : float, tuple of floats or list of floats, optional
        Parallactic angle threshold, expressed in FWHM, used to build the PCA
        library. If a tuple of 2 floats is provided, they are used as the lower
        and upper bounds of a linearly increasing threshold as a function of
        separation. If a list is provided, it will correspond to the threshold
        to be adopted for each annulus (length should match number of annuli).
        Default is (0.1, 1), which excludes 0.1 FWHM for the innermost annulus
        up to 1 FWHM for the outermost annulus.
    delta_sep : float, tuple of floats or list of floats, optional
        The radial threshold in terms of the mean FWHM, used to build the PCA
        library (for ADI+mSDI data). If a tuple of 2 floats is provided, they
        are used as the lower and upper bounds of a linearly increasing
        threshold as a function of separation. If a list is provided, it will
        correspond to the threshold to be adopted for each annulus (length
        should match number of annuli). Default is (0.1, 1), which excludes 0.1
        FWHM for the innermost annulus up to 1 FWHM for the outermost annulus.
    ncomp : 'auto', int, tuple/1d numpy array of int, list, tuple of lists, opt
        How many PCs are used as a lower-dimensional subspace to project the
        target (sectors of) frames. Depends on the dimensionality of `cube`.

        * ADI and ADI+RDI (``cube`` is a 3d array): if a single integer is
        provided, then the same number of PCs will be subtracted at each
        separation (annulus). If a tuple is provided, then a different number
        of PCs will be used for each annulus (starting with the innermost
        one). If ``ncomp`` is set to ``auto`` then the number of PCs are
        calculated for each region/patch automatically. If a list of int is
        provided, several npc will be tried at once, but the same value of npc
        will be used for all annuli. If a tuple of lists of int is provided,
        the length of tuple should match the number of annuli and different sets
        of npc will be calculated simultaneously for each annulus, with the
        exact values of npc provided in the respective lists.

        * ADI or ADI+RDI (``cube`` is a 4d array): same input format allowed as
        above, but with a slightly different behaviour if ncomp is a list: if it
        has the same length as the number of channels, each element of the list
        will be used as ``ncomp`` value (whether int, float or tuple) for each
        spectral channel. Otherwise the same behaviour as above is assumed.

        * ADI+mSDI case: ``ncomp`` must be a tuple of two integers or a list of
        tuples of two integers, with the number of PCs obtained from each
        multi-spectral frame (for each sector) and the number of PCs used in the
        second PCA stage (ADI fashion, using the residuals of the first stage).
        If None then the second PCA stage is skipped and the residuals are
        de-rotated and combined.

    svd_mode : Enum, see `vip_hci.config.paramenum.SvdMode`
        Switch for the SVD method/library to be used.
    nproc : None or int, optional
        Number of processes for parallel computing. If None the number of
        processes will be set to (cpu_count()/2).
    min_frames_lib : int, optional
        Minimum number of frames in the PCA reference library.
    max_frames_lib : int, optional
        Maximum number of frames in the PCA reference library. The more
        distant/decorrelated frames are removed from the library.
    tol : float, optional
        Stopping criterion for choosing the number of PCs when ``ncomp``
        is None. Lower values will lead to smaller residuals and more PCs.
    scaling : Enum, see `vip_hci.config.paramenum.Scaling`
        Pixel-wise scaling mode using ``sklearn.preprocessing.scale``
        function. If set to None, the input matrix is left untouched.
    imlib : Enum, see `vip_hci.config.paramenum.Imlib`
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation :  Enum, see `vip_hci.config.paramenum.Interpolation`
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : Enum, see `vip_hci.config.paramenum.Collapse`
        Sets the way of collapsing the frames for producing a final image.
    collapse_ifs : Enum, see `vip_hci.config.paramenum.Collapse`
        Sets how spectral residual frames should be combined to produce an
        mSDI image.
    ifs_collapse_range: str 'all' or tuple of 2 int
        If a tuple, it should contain the first and last channels where the mSDI
        residual channels will be collapsed (by default collapses all channels).
    full_output: boolean, optional
        Whether to return the final median combined image only or with other
        intermediate arrays.
    verbose : bool, optional
        If True prints to stdout intermediate info.
    theta_init : int
        Initial azimuth [degrees] of the first segment, counting from the
        positive x-axis counterclockwise (irrelevant if n_segments=1).
    weights: 1d numpy array or list, optional
        Weights to be applied for a weighted mean. Need to be provided if
        collapse mode is 'wmean'.
    cube_sig: numpy ndarray, opt
        Cube with estimate of significant authentic signals. If provided, this
        will be subtracted before projecting cube onto reference cube.

    Returns
    -------
    frame : numpy ndarray, 2d
        [full_output=False] Median combination of the de-rotated cube.
    array_out : numpy ndarray, 3d or 4d
        [full_output=True] Cube of residuals.
    array_der : numpy ndarray, 3d or 4d
        [full_output=True] Cube residuals after de-rotation.
    frame : numpy ndarray, 2d
        [full_output=True] Median combination of the de-rotated cube.
    """
    # Separate parameters of the ParamsObject from the optionnal rot_options
    class_params, rot_options = separate_kwargs_dict(all_kwargs,
                                                     PCA_ANNULAR_Params
                                                     )

    # Extracting the object of parameters (if any)
    algo_params = None
    if ALGO_KEY in rot_options.keys():
        algo_params = rot_options[ALGO_KEY]
        del rot_options[ALGO_KEY]

    if algo_params is None:
        algo_params = PCA_ANNULAR_Params(*all_args, **class_params)

    # by default, interpolate masked area before derotation if a mask is used
    if algo_params.radius_int and len(rot_options) == 0:
        rot_options['mask_val'] = 0
        rot_options['ker'] = 1
        rot_options['interp_zeros'] = True

    global start_time
    if algo_params.verbose:
        start_time = time_ini()

    if algo_params.left_eigv:
        if (
            (algo_params.cube_ref is not None)
            or (algo_params.cube_sig is not None)
            or (algo_params.ncomp == "auto")
        ):
            raise NotImplementedError(
                "left_eigv is not compatible"
                "with 'cube_ref', 'cube_sig', ncomp='auto'"
            )

    # ADI or ADI+RDI data
    if algo_params.cube.ndim == 3:
        if algo_params.verbose:
            add_params = {"start_time": start_time, "full_output": True}
        else:
            add_params = {"full_output": True}

        func_params = setup_parameters(
            params_obj=algo_params, fkt=_pca_adi_rdi, **add_params
        )
        res = _pca_adi_rdi(**func_params, **rot_options)

        cube_out, cube_der, frame = res
        if algo_params.full_output:
            return cube_out, cube_der, frame
        else:
            return frame
    else:
        raise ValueError("Wrong cube dimensions")


def _pca_adi_rdi(
    cube,
    angle_list,
    radius_int=0,
    fwhm=4,
    asize=2,
    n_segments=1,
    delta_rot=1,
    ncomp=1,
    svd_mode="lapack",
    nproc=None,
    min_frames_lib=2,
    max_frames_lib=200,
    tol=1e-1,
    scaling=None,
    imlib="opencv",
    interpolation="lanczos4",
    collapse="median",
    full_output=False,
    verbose=1,
    cube_ref=None,
    theta_init=0,
    weights=None,
    cube_sig=None,
    left_eigv=False,
    **rot_options,
):
    """PCA exploiting angular variability (ADI fashion)."""
    array = cube
    if array.ndim != 3:
        raise TypeError("Input array is not a cube or 3d array")
    if array.shape[0] != angle_list.shape[0]:
        raise TypeError("Input vector or parallactic angles has wrong length")

    n, y, x = array.shape

    store_nans = np.isnan(array)
    array[store_nans] = 0
    
    angle_list = check_pa_vector(angle_list)
    n_annuli = int((y / 2 - radius_int) / asize)

    if isinstance(delta_rot, tuple):
        delta_rot = np.linspace(delta_rot[0], delta_rot[1], num=n_annuli)
    elif np.isscalar(delta_rot):
        delta_rot = [delta_rot] * n_annuli
    else:
        if len(delta_rot) != n_annuli:
            msg = "If delta_rot is a list it should have n_annuli elements."
            raise TypeError(msg)

    if isinstance(n_segments, int):
        n_segments = [n_segments for _ in range(n_annuli)]
    elif n_segments == "auto":
        n_segments = list()
        n_segments.append(2)  # for first annulus
        n_segments.append(3)  # for second annulus
        ld = 2 * np.tan(360 / 4 / 2) * asize
        for i in range(2, n_annuli):  # rest of annuli
            radius = i * asize
            ang = np.rad2deg(2 * np.arctan(ld / (2 * radius)))
            n_segments.append(int(np.ceil(360 / ang)))

    if verbose:
        msg = "N annuli = {}, FWHM = {:.3f}"
        print(msg.format(n_annuli, fwhm))
        print("PCA per annulus (or annular sectors):")

    if nproc is None:  # Hyper-threading "duplicates" the cores -> cpu_count/2
        nproc = cpu_count() // 2

    # The annuli are built, and the corresponding PA thresholds for frame
    # rejection are calculated (at the center of the annulus)
    cube_out = np.zeros_like(array)
    if isinstance(ncomp, list):
        nncomp = len(ncomp)
        cube_out = np.zeros([nncomp, array.shape[0], array.shape[1],
                             array.shape[2]])
    if verbose:
        #  verbosity set to 2 only for ADI
        verbose_ann = int(verbose) + int(cube_ref is None)
    else:
        verbose_ann = verbose

    for ann in range(n_annuli):
        if isinstance(ncomp, tuple) or isinstance(ncomp, np.ndarray):
            if len(ncomp) == n_annuli:
                ncompann = ncomp[ann]
            else:
                msg = "If `ncomp` is a tuple, its length must match the number "
                msg += "of annuli"
                raise TypeError(msg)
        else:
            ncompann = ncomp

        n_segments_ann = n_segments[ann]
        res_ann_par = _define_annuli(
            angle_list,
            ann,
            n_annuli,
            fwhm,
            radius_int,
            asize,
            delta_rot[ann],
            n_segments_ann,
            verbose_ann,
            True,
        )
        pa_thr, inner_radius, ann_center = res_ann_par
        indices = get_annulus_segments(
            array[0], inner_radius, asize, n_segments_ann, theta_init
        )

        if left_eigv:
            indices_out = get_annulus_segments(array[0], inner_radius, asize,
                                               n_segments_ann, theta_init,
                                               out=True)

        # Library matrix is created for each segment and scaled if needed
        for j in range(n_segments_ann):
            yy = indices[j][0]
            xx = indices[j][1]
            matrix_segm = array[:, yy, xx]  # shape [nframes x npx_segment]
            matrix_segm = matrix_scaling(matrix_segm, scaling)
            if cube_ref is not None:
                matrix_segm_ref = cube_ref[:, yy, xx]
                matrix_segm_ref = matrix_scaling(matrix_segm_ref, scaling)
            else:
                matrix_segm_ref = None
            if cube_sig is not None:
                matrix_sig_segm = cube_sig[:, yy, xx]
            else:
                matrix_sig_segm = None

            if not left_eigv:
                res = pool_map(
                    nproc,
                    do_pca_patch,
                    matrix_segm,
                    iterable(range(n)),
                    angle_list,
                    fwhm,
                    pa_thr,
                    ann_center,
                    svd_mode,
                    ncompann,
                    min_frames_lib,
                    max_frames_lib,
                    tol,
                    matrix_segm_ref,
                    matrix_sig_segm,
                )
                if isinstance(ncomp, list):
                    nncomp = len(ncomp)
                    residuals = []
                    for nn in range(nncomp):
                        tmp = np.array([res[i][0][nn] for i in range(n)])
                        residuals.append(tmp)
                else:
                    res = np.array(res, dtype=object)
                    residuals = np.array(res[:, 0])
                    ncomps = res[:, 1]
                    nfrslib = res[:, 2]
            else:
                yy_out = indices_out[j][0]
                xx_out = indices_out[j][1]
                matrix_out_segm = array[
                    :, yy_out, xx_out
                ]  # shape [nframes x npx_out_segment]
                matrix_out_segm = matrix_scaling(matrix_out_segm, scaling)
                if isinstance(ncomp, list):
                    npc = max(ncomp)
                else:
                    npc = ncomp
                V = get_eigenvectors(npc, matrix_out_segm, svd_mode,
                                     noise_error=tol, left_eigv=True)

                if isinstance(ncomp, list):
                    residuals = []
                    for nn, npc_tmp in enumerate(ncomp):
                        transformed = np.dot(V[:npc_tmp], matrix_segm)
                        reconstructed = np.dot(transformed.T, V[:npc_tmp])
                        residuals.append(matrix_segm - reconstructed.T)
                else:
                    transformed = np.dot(V, matrix_segm)
                    reconstructed = np.dot(transformed.T, V)
                    residuals = matrix_segm - reconstructed.T
                    nfrslib = matrix_out_segm.shape[0]

            if isinstance(ncomp, list):
                for nn, npc in enumerate(ncomp):
                    for fr in range(n):
                        cube_out[nn, fr][yy, xx] = residuals[nn][fr]
            else:
                for fr in range(n):
                    cube_out[fr][yy, xx] = residuals[fr]

            # number of frames in library printed for each annular quadrant
            # number of PCs printed for each annular quadrant
            if verbose == 2 and not isinstance(ncomp, list):
                descriptive_stats(nfrslib, verbose=verbose, label="\tLIBsize: ")
                descriptive_stats(ncomps, verbose=verbose, label="\tNum PCs: ")

        if verbose == 1:
            print("Done PCA with {} for current annulus".format(svd_mode))
            timing(start_time)

    cube_out[store_nans] = np.nan

    if isinstance(ncomp, list):
        cube_der = np.zeros_like(cube_out)
        frame = []
        for nn, npc in enumerate(ncomp):
            cube_der[nn] = cube_derotate(cube_out[nn], angle_list, nproc=nproc,
                                         imlib=imlib,
                                         interpolation=interpolation,
                                         **rot_options)
            frame.append(cube_collapse(cube_der[nn], mode=collapse, w=weights))
    else:
        # Cube is derotated according to the parallactic angle and collapsed
        origin = np.asarray([np.shape(cube_out)[1] / 2 - 0.5, np.shape(cube_out)[2] / 2 - 0.5])
        cube_der = rotate_imgs(cube_out, angle_list, np.array(len(cube_out)*[origin]), numthreads=nproc, flipx=False, new_center=origin)
    
        '''
        cube_der = cube_derotate(
            cube_out,
            angle_list,
            nproc=nproc,
            imlib=imlib,
            interpolation=interpolation,
            **rot_options,
        )
        '''
        
        frame = cube_collapse(cube_der, mode=collapse, w=weights)

    if verbose:
        print("Done derotating and combining.")
        timing(start_time)

    if full_output:
        return cube_out, cube_der, frame
    else:
        return frame

def do_pca_patch(
    matrix,
    frame,
    angle_list,
    fwhm,
    pa_threshold,
    ann_center,
    svd_mode,
    ncomp,
    min_frames_lib,
    max_frames_lib,
    tol,
    matrix_ref,
    matrix_sig_segm,
):
    """Do the SVD/PCA for each frame patch (small matrix).

    For each frame, frames to be rejected from the PCA library are found
    depending on the criterion in field rotation. The library is also truncated
    on the other end (frames too far in time, which have rotated more) which are
    more decorrelated, to keep the computational cost lower. This truncation is
    done on the annuli beyong 10*FWHM radius and the goal is to keep
    min(num_frames/2, 200) in the library.

    """
    if pa_threshold != 0:
        indices_left = _find_indices_adi(angle_list, frame, pa_threshold,
                                         truncate=True,
                                         max_frames=max_frames_lib)
        msg = "Too few frames left in the PCA library. "
        msg += "Accepted indices length ({:.0f}) less than {:.0f}. "
        msg += "Try decreasing either delta_rot or min_frames_lib."
        try:
            if matrix_sig_segm is not None:
                data_ref = matrix[indices_left] - matrix_sig_segm[indices_left]
            else:
                data_ref = matrix[indices_left]
        except IndexError:
            if matrix_ref is None:
                raise RuntimeError(msg.format(0, min_frames_lib))
            data_ref = None

        if data_ref.shape[0] < min_frames_lib and matrix_ref is None:
            raise RuntimeError(msg.format(len(indices_left), min_frames_lib))
    else:
        if matrix_sig_segm is not None:
            data_ref = matrix - matrix_sig_segm
        else:
            data_ref = matrix

    if matrix_ref is not None:
        # Stacking the ref and the target ref (pa thresh) libraries
        if data_ref is not None:
            data_ref = np.vstack((matrix_ref, data_ref))
        else:
            data_ref = matrix_ref

    curr_frame = matrix[frame]  # current frame
    if matrix_sig_segm is not None:
        curr_frame_emp = matrix[frame] - matrix_sig_segm[frame]
    else:
        curr_frame_emp = curr_frame
    if isinstance(ncomp, list):
        npc = max(ncomp)
    else:
        npc = ncomp
    V = get_eigenvectors(npc, data_ref, svd_mode, noise_error=tol)

    if isinstance(ncomp, list):
        residuals = []
        for nn, npc_tmp in enumerate(ncomp):
            transformed = np.dot(curr_frame_emp, V[:npc_tmp].T)
            reconstructed = np.dot(transformed.T, V[:npc_tmp])
            residuals.append(curr_frame - reconstructed)
    else:
        transformed = np.dot(curr_frame_emp, V.T)
        reconstructed = np.dot(transformed.T, V)
        residuals = curr_frame - reconstructed

    return residuals, V.shape[0], data_ref.shape[0]



def pca(*all_args: List, **all_kwargs: dict):
    """Full-frame PCA algorithm applied to PSF subtraction.

    The reference PSF and the quasi-static speckle pattern are modeled using
    Principal Component Analysis. Depending on the input parameters this PCA
    function can work in ADI, RDI or mSDI (IFS data) mode.

    ADI: the target ``cube`` itself is used to learn the PCs and to obtain a
    low-rank approximation model PSF (star + speckles). Both `cube_ref`` and
    ``scale_list`` must be None. The full-frame ADI-PCA implementation is based
    on [AMA12]_ and [SOU12]_. If ``batch`` is provided then the cube is
    processed with incremental PCA as described in [GOM17]_.

    (ADI+)RDI: if a reference cube is provided (``cube_ref``), its PCs are used
    to reconstruct the target frames to obtain the model PSF (star + speckles).

    (ADI+)mSDI (IFS data): if a scaling vector is provided (``scale_list``) and
    the cube is a 4d array [# channels, # adi-frames, Y, X], it's assumed it
    contains several multi-spectral frames acquired in pupil-stabilized mode.
    A single or two stages PCA can be performed, depending on ``adimsdi``, as
    explained in [CHR19]_.

    Parameters
    ----------
    all_args: list, optional
        Positional arguments for the PCA algorithm. Full list of parameters
        below.
    all_kwargs: dictionary, optional
        Mix of keyword arguments that can initialize a PCA_Params and the
        optional 'rot_options' dictionary (with keyword values ``border_mode``,
        ``mask_val``, ``edge_blend``, ``interp_zeros``, ``ker``; see docstring
        of ``vip_hci.preproc.frame_rotate``). Can also contain a PCA_Params
        dictionary named `algo_params`.

    PCA parameters
    --------------
    cube : str or numpy ndarray, 3d or 4d
        Input cube (ADI or ADI+mSDI). If 4D, the first dimension should be
        spectral. If a string is given, it must correspond to the path to the
        fits file to be opened in memmap mode (incremental PCA-ADI of 3D cubes
        only).
    angle_list : numpy ndarray, 1d
        Vector of derotation angles to align North up in your images.
    cube_ref : 3d or 4d numpy ndarray, or list of 3D numpy ndarray, optional
        Reference library cube for Reference Star Differential Imaging. Should
        be 3D, except if input cube is 4D and no scale_list is provided,
        reference cube can then either be 4D or a list of 3D cubes (i.e.
        providing the reference cube for each individual spectral cube).
    scale_list : numpy ndarray, 1d, optional
        If provided, triggers mSDI reduction. These should be the scaling
        factors used to re-scale the spectral channels and align the speckles
        in case of IFS data (ADI+mSDI cube). Usually, the
        scaling factors are the last channel wavelength divided by the
        other wavelengths in the cube (more thorough approaches can be used
        to get the scaling factors, e.g. with
        ``vip_hci.preproc.find_scal_vector``).
    ncomp : int, float, tuple of int/None, or list, optional
        How many PCs are used as a lower-dimensional subspace to project the
        target frames.

        * ADI (``cube`` is a 3d array): if an int is provided, ``ncomp`` is the
        number of PCs extracted from ``cube`` itself. If ``ncomp`` is a float
        in the interval [0, 1] then it corresponds to the desired cumulative
        explained variance ratio (the corresponding number of components is
        estimated). If ``ncomp`` is a tuple of two integers, then it
        corresponds to an interval of PCs in which final residual frames are
        computed (optionally, if a tuple of 3 integers is passed, the third
        value is the step). If ``ncomp`` is a list of int, these will be used to
        calculate residual frames. When ``ncomp`` is a tuple or list, and
        ``source_xy`` is not None, then the S/Ns (mean value in a 1xFWHM
        circular aperture) of the given (X,Y) coordinates are computed.

        * ADI+RDI (``cube`` and ``cube_ref`` are 3d arrays): ``ncomp`` is the
        number of PCs obtained from ``cube_ref``. If ``ncomp`` is a tuple,
        then it corresponds to an interval of PCs (obtained from ``cube_ref``)
        in which final residual frames are computed. If ``ncomp`` is a list of
        int, these will be used to calculate residual frames. When ``ncomp`` is
        a tuple or list, and ``source_xy`` is not None, then the S/Ns (mean
        value in a 1xFWHM circular aperture) of the given (X,Y) coordinates are
        computed.

        * ADI or ADI+RDI (``cube`` is a 4d array): same input format allowed as
        above. If ``ncomp`` is a list with the same length as the number of
        channels, each element of the list will be used as ``ncomp`` value
        (be it int, float or tuple) for each spectral channel. If not a
        list or a list with a different length as the number of spectral
        channels, these will be tested for all spectral channels respectively.

        * ADI+mSDI (``cube`` is a 4d array and ``adimsdi="single"``): ``ncomp``
        is the number of PCs obtained from the whole set of frames
        (n_channels * n_adiframes). If ``ncomp`` is a float in the interval
        (0, 1] then it corresponds to the desired CEVR, and the corresponding
        number of components will be estimated. If ``ncomp`` is a tuple, then
        it corresponds to an interval of PCs in which final residual frames
        are computed. If ``ncomp`` is a list of int, these will be used to
        calculate residual frames. When ``ncomp`` is a tuple or list, and
        ``source_xy`` is not None, then the S/Ns (mean value in a 1xFWHM
        circular aperture) of the given (X,Y) coordinates are computed.

        * ADI+mSDI  (``cube`` is a 4d array and ``adimsdi="double"``): ``ncomp``
        must be a tuple, where the first value is the number of PCs obtained
        from each multi-spectral frame (if None then this stage will be
        skipped and the spectral channels will be combined without
        subtraction); the second value sets the number of PCs used in the
        second PCA stage, ADI-like using the residuals of the first stage (if
        None then the second PCA stage is skipped and the residuals are
        de-rotated and combined).

    svd_mode : Enum, see `vip_hci.config.paramenum.SvdMode`
        Switch for the SVD method/library to be used.
    scaling : Enum, or tuple of Enum, see `vip_hci.config.paramenum.Scaling`
        Pixel-wise scaling mode using ``sklearn.preprocessing.scale``
        function. If set to None, the input matrix is left untouched. In the
        case of PCA-SADI in 2 steps, this can be a tuple of 2 values,
        corresponding to the scaling for each of the 2 steps of PCA.
    mask_center_px : None or int
        If None, no masking is done. If an integer > 1 then this value is the
        radius of the circular mask.
    source_xy : tuple of int, optional
        For ADI-PCA, this triggers a frame rejection in the PCA library, with
        ``source_xy`` as the coordinates X,Y of the center of the annulus where
        the PA criterion is estimated. When ``ncomp`` is a tuple, a PCA grid is
        computed and the S/Ns (mean value in a 1xFWHM circular aperture) of the
        given (X,Y) coordinates are computed.
    delta_rot : int, optional
        Factor for tuning the parallactic angle threshold, expressed in FWHM.
        Default is 1 (excludes 1xFWHM on each side of the considered frame).
    fwhm : float, list or 1d numpy array, optional
        Known size of the FWHM in pixels to be used. Default value is 4.
        Can be a list or 1d numpy array for a 4d input cube with no scale_list.
    adimsdi : Enum, see `vip_hci.config.paramenum.Adimsdi`
        Changes the way the 4d cubes (ADI+mSDI) are processed. Basically it
        determines whether a single or double pass PCA is going to be computed.
    crop_ifs: bool, optional
        [adimsdi='single'] If True cube is cropped at the moment of frame
        rescaling in wavelength. This is recommended for large FOVs such as the
        one of SPHERE, but can remove significant amount of information close to
        the edge of small FOVs (e.g. SINFONI).
    imlib : Enum, see `vip_hci.config.paramenum.Imlib`
        See the documentation of ``vip_hci.preproc.frame_rotate``.
    imlib2 : Enum, see `vip_hci.config.paramenum.Imlib`
        See the documentation of ``vip_hci.preproc.cube_rescaling_wavelengths``.
    interpolation : Enum, see `vip_hci.config.paramenum.Interpolation`
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : Enum, see `vip_hci.config.paramenum.Collapse`
        Sets how temporal residual frames should be combined to produce an
        ADI image.
    collapse_ifs : Enum, see `vip_hci.config.paramenum.Collapse`
        Sets how spectral residual frames should be combined to produce an
        mSDI image.
    ifs_collapse_range: str 'all' or tuple of 2 int
        If a tuple, it should contain the first and last channels where the mSDI
        residual channels will be collapsed (by default collapses all channels).
    smooth: float or None, optional
        Gaussian kernel size to use to smooth the images. None by default (no
        smoothing). Can be used when pca is used within NEGFC with the Hessian
        figure of merit.
    smooth_first_pass: float or None, optional
        [adimsdi='double'] For 4D cubes with requested PCA-SADI processing in 2
        steps, the Gaussian kernel size to use to smooth the images of the first
        pass before performing the second pass. None by default (no smoothing).
    mask_rdi: tuple of two numpy array or one signle 2d numpy array, opt
        If provided, binary mask(s) will be used either in RDI mode or in
        ADI+mSDI (2 steps) mode. If two masks are provided, they will the anchor
        and boat regions, respectively, following the denominations in [REN23]_.
        If only one mask is provided, it will be used as the anchor, and the
        boat images will not be masked (i.e., full frames used).
    ref_strategy: str, opt {'RDI', 'ARDI', 'RSDI', 'ARSDI'}
        [cube_ref is not None] Indicates the strategy to be adopted when a
        reference cube is provided. By default, RDI is done for a 3D input cube,
        while RSDI is done for a 4D input cube if a ``scale_list`` is provided
        (otherwise RDI is done channel per channel). RSDI rescales all channels
        to build a larger reference library available for each channel. If
        ``ref_strategy`` is set to 'ARDI' or 'ARSDI', the PCA library is made of
        both the science and reference images. In the case of 'ARSDI', all
        channels (science and reference) are rescaled for a larger library.
    check_memory : bool, optional
        If True, it checks that the input cube is smaller than the available
        system memory.
    batch : None, int or float, optional
        When it is not None, it triggers the incremental PCA (for ADI and
        ADI+mSDI cubes). If an int is given, it corresponds to the number of
        frames in each sequential mini-batch. If a float (0, 1] is given, it
        corresponds to the size of the batch is computed wrt the available
        memory in the system.
    nproc : None or int, optional
        Number of processes for parallel computing. If None the number of
        processes will be set to (cpu_count()/2). Defaults to ``nproc=1``.
    full_output: bool, optional
        Whether to return the final median combined image only or with other
        intermediate arrays.
    verbose : bool, optional
        If True prints intermediate info and timing.
    weights: 1d numpy array or list, optional
        Weights to be applied for a weighted mean. Need to be provided if
        collapse mode is 'wmean'.
    left_eigv : bool, optional
        Whether to use rather left or right singularvectors
        This mode is not compatible with 'mask_rdi' and 'batch'
    min_frames_pca : int, optional
        Minimum number of frames required in the PCA library. An error is raised
        if less than such number of frames can be found.
    cube_sig: numpy ndarray, opt
        Cube with estimate of significant authentic signals. If provided, this
        will be subtracted before projecting considering the science cube as
        reference cube.
    med_of_npcs: bool, opt
        [ncomp is tuple or list] Whether to consider the median image of the
        list of images obtained with a list or tuple of ncomp values.

    Return
    -------
    final_residuals_cube : List of numpy ndarray
        [(ncomp is tuple or list) & (med_of_npcs=False or source_xy != None)]
        List of residual final PCA frames obtained for a grid of PC values.
    frame : numpy ndarray
        [(ncomp is scalar) or (source_xy != None)] 2D array, median combination
        of the de-rotated/re-scaled residuals cube.
        [(ncomp is tuple or list) & (med_of_npcs=True)] median of images
        obtained with different ncomp values.
    pcs : numpy ndarray
        [full_output=True, source_xy=None] Principal components. Valid for
        ADI cubes 3D or 4D (i.e. ``scale_list=None``). This is also returned
        when ``batch`` is not None (incremental PCA).
    recon_cube, recon : numpy ndarray
        [full_output=True] Reconstructed cube. Valid for ADI cubes 3D or 4D
        (i.e. ``scale_list=None``)
    residuals_cube : numpy ndarray
        [full_output=True] Residuals cube. Valid for ADI cubes 3D or 4D
        (i.e. ``scale_list=None``)
    residuals_cube_ : numpy ndarray
        [full_output=True] Derotated residuals cube. Valid for ADI cubes 3D or
        4D (i.e. ``scale_list=None``)
    residuals_cube_channels : numpy ndarray
        [full_output=True, adimsdi='double'] Residuals for each multi-spectral
        cube. Valid for ADI+mSDI (4D) cubes (when ``scale_list`` is provided)
    residuals_cube_channels_ : numpy ndarray
        [full_output=True, adimsdi='double'] Derotated final residuals. Valid
        for ADI+mSDI (4D) cubes (when ``scale_list`` is provided)
    cube_allfr_residuals : numpy ndarray
        [full_output=True, adimsdi='single']  Residuals cube (of the big cube
        with channels and time processed together). Valid for ADI+mSDI (4D)
        cubes (when ``scale_list`` is provided)
    cube_desc_residuals : numpy ndarray
        [full_output=True, adimsdi='single'] Residuals cube after de-scaling the
        spectral frames to their original scale. Valid for ADI+mSDI (4D) (when
        ``scale_list`` is provided).
    cube_adi_residuals : numpy ndarray
        [full_output=True, adimsdi='single'] Residuals cube after de-scaling the
        spectral frames to their original scale and collapsing the channels.
        Valid for ADI+mSDI (4D) (when ``scale_list`` is provided).
    ifs_adi_frames : numpy ndarray
        [full_output=True, 4D input cube, ``scale_list=None``] This is the cube
        of individual ADI reductions for each channel of the IFS cube.
    medians : numpy ndarray
        [full_output=True, source_xy=None, batch!=None] Median images of each
        batch, in incremental PCA, for 3D input cubes only.

    """
    # Separating the parameters of the ParamsObject from optional rot_options

    class_params, rot_options = separate_kwargs_dict(
        initial_kwargs=all_kwargs, parent_class=PCA_Params
    )

    # Extracting the object of parameters (if any)
    algo_params = None
    if ALGO_KEY in rot_options.keys():
        algo_params = rot_options[ALGO_KEY]
        del rot_options[ALGO_KEY]

    if algo_params is None:
        algo_params = PCA_Params(*all_args, **class_params)

    # by default, interpolate masked area before derotation if a mask is used
    if algo_params.mask_center_px and len(rot_options) == 0:
        rot_options['mask_val'] = 0
        rot_options['ker'] = 1
        rot_options['interp_zeros'] = True

    start_time = time_ini(algo_params.verbose)

    if algo_params.batch is None:
        check_array(algo_params.cube, (3, 4), msg="cube")
    else:
        if not isinstance(algo_params.cube, (str, np.ndarray)):
            raise TypeError(
                "`cube` must be a numpy (3d or 4d) array or a str "
                "with the full path on disk"
            )

    if algo_params.left_eigv:
        if (
            algo_params.batch is not None
            or algo_params.mask_rdi is not None
            or algo_params.cube_ref is not None
        ):
            raise NotImplementedError(
                "left_eigv is not compatible with 'mask_rdi' nor 'batch'"
            )

    # checking memory (if in-memory numpy array is provided)
    if not isinstance(algo_params.cube, str):
        input_bytes = (
            algo_params.cube_ref.nbytes
            if algo_params.cube_ref is not None
            else algo_params.cube.nbytes
        )
        mem_msg = (
            "Set check_memory=False to override this memory check or "
            "set `batch` to run incremental PCA (valid for ADI or "
            "ADI+mSDI single-pass)"
        )
        check_enough_memory(
            input_bytes,
            1.0,
            raise_error=algo_params.check_memory,
            error_msg=mem_msg,
            verbose=algo_params.verbose,
        )

    if algo_params.nproc is None:
        algo_params.nproc = cpu_count() // 2  # Hyper-threading doubles # cores

    # All possible outputs for any PCA usage must be pre-declared to None
    # Default possible outputs

    (
        frame,
        final_residuals_cube,
        pclist,
        pcs,
        medians,
        recon,
        residuals_cube,
        residuals_cube_,
    ) = (None for _ in range(8))

    # Full_output/cube dimension dependant variables

    (
        table,
        cube_allfr_residuals,
        cube_adi_residuals,
        residuals_cube_channels,
        residuals_cube_channels_,
        ifs_adi_frames,
    ) = (None for _ in range(6))

    # ADI + mSDI. Shape of cube: (n_channels, n_adi_frames, y, x)
    # isinstance(cube, np.ndarray) and cube.ndim == 4:
    if algo_params.scale_list is not None:
        add_params = {"start_time": start_time}
        if algo_params.cube_ref is not None:
            if algo_params.cube_ref.ndim != 4:
                msg = "Ref cube has wrong format for 4d input cube"
                raise TypeError(msg)
            if 'A' in algo_params.ref_strategy:  # e.g. 'ARSDI'
                add_params["ref_strategy"] = 'ARSDI'  # uniformize
                if algo_params.adimsdi == Adimsdi.SINGLE:
                    cube_ref = np.concatenate((algo_params.cube,
                                               algo_params.cube_ref), axis=1)
                    add_params["cube_ref"] = cube_ref
            else:
                add_params["ref_strategy"] = 'RSDI'

        if algo_params.adimsdi == Adimsdi.DOUBLE:
            func_params = setup_parameters(
                params_obj=algo_params, fkt=_adimsdi_doublepca, **add_params
            )
            res_pca = _adimsdi_doublepca(
                **func_params,
                **rot_options,
            )
            residuals_cube_channels, residuals_cube_channels_, frame = res_pca
        elif algo_params.adimsdi == Adimsdi.SINGLE:
            func_params = setup_parameters(
                params_obj=algo_params, fkt=_adimsdi_singlepca, **add_params
            )
            res_pca = _adimsdi_singlepca(
                **func_params,
                **rot_options,
            )
            if np.isscalar(algo_params.ncomp):
                cube_allfr_residuals, cube_desc_residuals = res_pca[:2]
                cube_adi_residuals, frame = res_pca[2:]
            elif isinstance(algo_params.ncomp, (tuple, list)):
                if algo_params.source_xy is None:
                    if algo_params.full_output:
                        final_residuals_cube, pclist = res_pca
                    else:
                        final_residuals_cube = res_pca
                else:
                    final_residuals_cube, frame, table, _ = res_pca
        else:
            raise ValueError("`adimsdi` mode not recognized")

    # 4D cube, but no mSDI desired
    elif algo_params.cube.ndim == 4:
        nch, nz, ny, nx = algo_params.cube.shape
        ifs_adi_frames = np.zeros([nch, ny, nx])
        if not isinstance(algo_params.ncomp, list):
            ncomp = [algo_params.ncomp] * nch
        elif len(algo_params.ncomp) != nch:
            nnpc = len(algo_params.ncomp)
            ifs_adi_frames = np.zeros([nch, nnpc, ny, nx])
            ncomp = [algo_params.ncomp] * nch
        else:
            ncomp = algo_params.ncomp
        if np.isscalar(algo_params.fwhm):
            algo_params.fwhm = [algo_params.fwhm] * nch

        pcs = []
        recon = []
        residuals_cube = []
        residuals_cube_ = []
        final_residuals_cube = []
        recon_cube = []
        medians = []
        table = []
        pclist = []
        grid_case = False

        # ADI or RDI
        for ch in range(nch):
            add_params = {
                "start_time": start_time,
                "cube": algo_params.cube[ch],
                "ncomp": ncomp[ch],  # algo_params.ncomp[ch],
                "fwhm": algo_params.fwhm[ch],
                "full_output": True,
            }

            # RDI
            if algo_params.cube_ref is not None:
                if algo_params.cube_ref[ch].ndim != 3:
                    msg = "Ref cube has wrong format for 4d input cube"
                    raise TypeError(msg)
                if algo_params.ref_strategy == 'RDI':
                    add_params["cube_ref"] = algo_params.cube_ref[ch]
                elif algo_params.ref_strategy == 'ARDI':
                    cube_ref = np.concatenate((algo_params.cube[ch],
                                               algo_params.cube_ref[ch]))
                    add_params["cube_ref"] = cube_ref
                else:
                    msg = "ref_strategy argument not recognized."
                    msg += "Should be 'RDI' or 'ARDI'"
                    raise TypeError(msg)

            func_params = setup_parameters(
                params_obj=algo_params, fkt=_adi_rdi_pca, **add_params
            )
            res_pca = _adi_rdi_pca(
                **func_params,
                **rot_options,
            )

            if algo_params.batch is None:
                if algo_params.source_xy is not None:
                    # PCA grid, computing S/Ns
                    if isinstance(ncomp[ch], (tuple, list)):
                        final_residuals_cube.append(res_pca[0])
                        ifs_adi_frames[ch] = res_pca[1]
                        table.append(res_pca[2])
                    # full-frame PCA with rotation threshold
                    else:
                        recon_cube.append(res_pca[0])
                        residuals_cube.append(res_pca[1])
                        residuals_cube_.append(res_pca[2])
                        ifs_adi_frames[ch] = res_pca[-1]
                else:
                    # PCA grid
                    if isinstance(ncomp[ch], (tuple, list)):
                        ifs_adi_frames[ch] = res_pca[0]
                        pclist.append(res_pca[1])
                        grid_case = True
                    # full-frame standard PCA
                    else:
                        pcs.append(res_pca[0])
                        recon.append(res_pca[1])
                        residuals_cube.append(res_pca[2])
                        residuals_cube_.append(res_pca[3])
                        ifs_adi_frames[ch] = res_pca[-1]
            # full-frame incremental PCA
            else:
                ifs_adi_frames[ch] = res_pca[0]
                pcs.append(res_pca[2])
                medians.append(res_pca[3])

        if grid_case:
            for i in range(len(ncomp[0])):
                frame = cube_collapse(ifs_adi_frames[:, i],
                                      mode=algo_params.collapse_ifs)
                final_residuals_cube.append(frame)
        else:
            frame = cube_collapse(ifs_adi_frames,
                                  mode=algo_params.collapse_ifs)

        # convert to numpy arrays when relevant
        if len(pcs) > 0:
            pcs = np.array(pcs)
        if len(recon) > 0:
            recon = np.array(recon)
        if len(residuals_cube) > 0:
            residuals_cube = np.array(residuals_cube)
        if len(residuals_cube_) > 0:
            residuals_cube_ = np.array(residuals_cube_)
        if len(final_residuals_cube) > 0:
            final_residuals_cube = np.array(final_residuals_cube)
        if len(recon_cube) > 0:
            recon_cube = np.array(recon_cube)
        if len(medians) > 0:
            medians = np.array(medians)

    # 3D RDI or ADI. Shape of cube: (n_adi_frames, y, x)
    else:
        add_params = {
            "start_time": start_time,
            "full_output": True,
        }

        if algo_params.cube_ref is not None and algo_params.batch is not None:
            raise ValueError("RDI not compatible with batch mode")
        elif algo_params.cube_ref is not None:
            if algo_params.ref_strategy == 'ARDI':
                algo_params.cube_ref = np.concatenate((algo_params.cube,
                                                       algo_params.cube_ref))
            elif algo_params.ref_strategy != 'RDI':
                msg = "ref_strategy argument not recognized."
                msg += "Should be 'RDI' or 'ARDI'"
                raise TypeError(msg)

        func_params = setup_parameters(params_obj=algo_params,
                                       fkt=_adi_rdi_pca, **add_params)

        res_pca = _adi_rdi_pca(**func_params, **rot_options)

        if algo_params.batch is None:
            if algo_params.source_xy is not None:
                # PCA grid, computing S/Ns
                if isinstance(algo_params.ncomp, (tuple, list)):
                    if algo_params.full_output:
                        final_residuals_cube, frame, table, _ = res_pca
                    else:
                        # returning only the optimal residual
                        frame = res_pca[1]
                # full-frame PCA with rotation threshold
                else:
                    recon_cube, residuals_cube, residuals_cube_, frame = res_pca
            else:
                # PCA grid
                if isinstance(algo_params.ncomp, (tuple, list)):
                    final_residuals_cube, pclist = res_pca
                # full-frame standard PCA
                else:
                    pcs, recon, residuals_cube, residuals_cube_, frame = res_pca
        # full-frame incremental PCA
        else:
            frame, _, pcs, medians = res_pca

    # else:
    #     raise RuntimeError(
    #        "Only ADI, ADI+RDI and ADI+mSDI observing techniques are supported"
    #     )

    # --------------------------------------------------------------------------
    # Returns for each case (ADI, ADI+RDI and ADI+mSDI) and combination of
    # parameters: full_output, source_xy, batch, ncomp
    # --------------------------------------------------------------------------
    # If requested (except when source_xy is not None), return median image
    # cond_s = algo_params.source_xy is None
    if final_residuals_cube is not None and algo_params.med_of_npcs:
        final_residuals_cube = np.median(final_residuals_cube, axis=0)

    isarr = isinstance(algo_params.cube, np.ndarray)
    if isarr and algo_params.scale_list is not None:
        # ADI+mSDI double-pass PCA
        if algo_params.adimsdi == Adimsdi.DOUBLE:
            if algo_params.full_output:
                return frame, residuals_cube_channels, residuals_cube_channels_
            else:
                return frame

        elif algo_params.adimsdi == Adimsdi.SINGLE:
            # ADI+mSDI single-pass PCA
            if np.isscalar(algo_params.ncomp):
                if algo_params.full_output:
                    return (frame, cube_allfr_residuals, cube_desc_residuals,
                            cube_adi_residuals)
                else:
                    return frame
            # ADI+mSDI single-pass PCA grid
            elif isinstance(algo_params.ncomp, (tuple, list)):
                if algo_params.source_xy is None:
                    if algo_params.full_output:
                        return final_residuals_cube, pclist
                    else:
                        return final_residuals_cube
                else:
                    if algo_params.full_output:
                        return final_residuals_cube, frame, table
                    else:
                        return frame
            else:
                msg = "ncomp value should only be a float, an int or a tuple of"
                msg += f" those, not a {type(algo_params.ncomp)}."
                raise ValueError(msg)
        else:
            msg = f"ADIMSDI value should only be {Adimsdi.SINGLE} or"
            msg += f" {Adimsdi.DOUBLE}."
            raise ValueError(msg)

    # ADI and ADI+RDI (3D or 4D)
    elif isinstance(algo_params.cube, str) or algo_params.scale_list is None:
        if algo_params.source_xy is None and algo_params.full_output:
            # incremental PCA
            if algo_params.batch is not None:
                final_res = [frame, pcs, medians]
            else:
                # PCA grid
                if isinstance(algo_params.ncomp, (tuple, list)):
                    final_res = [final_residuals_cube, pclist]
                # full-frame standard PCA or ADI+RDI
                else:
                    final_res = [frame, pcs, recon, residuals_cube,
                                 residuals_cube_]
            if algo_params.cube.ndim == 4:
                final_res.append(ifs_adi_frames)
            return tuple(final_res)
        elif algo_params.source_xy is not None and algo_params.full_output:
            # PCA grid, computing S/Ns
            if isinstance(algo_params.ncomp, (tuple, list)):
                final_res = [final_residuals_cube, frame, table]
            # full-frame PCA with rotation threshold
            else:
                final_res = [frame, recon_cube, residuals_cube, residuals_cube_]
            if algo_params.cube.ndim == 4:
                final_res.append(ifs_adi_frames)
            return tuple(final_res)
        elif algo_params.source_xy is not None:
            return frame
        elif not algo_params.full_output:
            # PCA grid
            if isinstance(algo_params.ncomp, (tuple, list)):
                return final_residuals_cube
            # full-frame standard PCA or ADI+RDI
            else:
                return frame

    else:
        msg = "cube value should only be a str or a numpy.ndarray, not a "
        msg += f"{type(algo_params.cube)}."

def _adi_rdi_pca(
    cube,
    cube_ref,
    angle_list,
    ncomp,
    batch,
    source_xy,
    delta_rot,
    fwhm,
    scaling,
    mask_center_px,
    svd_mode,
    imlib,
    interpolation,
    collapse,
    verbose,
    start_time,
    nproc,
    full_output,
    weights=None,
    mask_rdi=None,
    cube_sig=None,
    left_eigv=False,
    min_frames_pca=10,
    max_frames_pca=None,
    smooth=None,
    **rot_options,
):
    """Handle the ADI or ADI+RDI PCA post-processing."""
    (
        frame,
        pcs,
        recon,
        residuals_cube,
        residuals_cube_,
    ) = (None for _ in range(5))
    # Full/Single ADI processing, incremental PCA
    if batch is not None:
        result = pca_incremental(
            cube,
            angle_list,
            batch=batch,
            ncomp=ncomp,
            collapse=collapse,
            verbose=verbose,
            full_output=full_output,
            start_time=start_time,
            weights=weights,
            nproc=nproc,
            imlib=imlib,
            interpolation=interpolation,
            **rot_options,
        )
        return result

    else:
        # Full/Single ADI processing
        n, y, x = cube.shape

        angle_list = check_pa_vector(angle_list)
        if not n == angle_list.shape[0]:
            raise ValueError(
                "`angle_list` vector has wrong length. It must "
                "equal the number of frames in the cube"
            )

        if not np.isscalar(ncomp) and not isinstance(ncomp, (tuple, list)):
            msg = "`ncomp` must be an int, float, tuple or list in the ADI case"
            raise TypeError(msg)

        if np.isscalar(ncomp):
            if cube_ref is not None:
                nref = cube_ref.shape[0]
            else:
                nref = n
            if isinstance(ncomp, int) and ncomp > nref:
                ncomp = min(ncomp, nref)
                print(
                    "Number of PCs too high (max PCs={}), using {} PCs "
                    "instead.".format(nref, ncomp)
                )
            elif ncomp <= 0:
                msg = "Number of PCs too low. It should be > 0."
                raise ValueError(msg)
            if mask_rdi is None:
                if source_xy is None:
                    store_nans = np.isnan(cube)
                    cube_nanless = np.copy(cube)
                    cube_nanless[store_nans] = 0
                    residuals_result = _project_subtract(
                        cube_nanless,
                        cube_ref,
                        ncomp,
                        scaling,
                        mask_center_px,
                        svd_mode,
                        verbose,
                        full_output,
                        cube_sig=cube_sig,
                        left_eigv=left_eigv,
                    )
                    if verbose:
                        timing(start_time)
                    if full_output:
                        residuals_cube = residuals_result[0]
                        reconstructed = residuals_result[1]
                        V = residuals_result[2]
                        pcs = reshape_matrix(V, y, x) if not left_eigv else V.T
                        recon = reshape_matrix(reconstructed, y, x)
                    else:
                        residuals_cube = residuals_result
                    residuals_cube[store_nans] = np.nan

            else:
                residuals_result = cube_subtract_sky_pca(
                    cube, cube_ref, mask_rdi, ncomp=ncomp, full_output=True
                )
                residuals_cube = residuals_result[0]
                pcs = residuals_result[2]
                recon = residuals_result[-1]

            # Cube is derotated according to the parallactic angle and collapsed
            origin = np.asarray([np.shape(residuals_cube)[1] / 2 - 0.5, np.shape(residuals_cube)[2] / 2 - 0.5])
            residuals_cube_ = rotate_imgs(residuals_cube, angle_list, np.array(len(residuals_cube)*[origin]), numthreads=nproc, flipx=False, new_center=origin)

            '''
            residuals_cube_ = cube_derotate(
                residuals_cube,
                angle_list,
                nproc=nproc,
                imlib=imlib,
                interpolation=interpolation,
                **rot_options,
            )
            '''
            frame = cube_collapse(residuals_cube_, mode=collapse, w=weights)
            
            if smooth is not None:
                frame = frame_filter_lowpass(frame, mode='gauss',
                                             fwhm_size=smooth)
            if mask_center_px:
                residuals_cube_ = mask_circle(residuals_cube_, mask_center_px)
                frame = mask_circle(frame, mask_center_px)
            if verbose:
                print("Done de-rotating and combining")
                timing(start_time)
            if source_xy is not None:
                if full_output:
                    return (recon_cube,
                            residuals_cube,
                            residuals_cube_,
                            frame)
                else:
                    return frame
            else:
                if full_output:
                    return (pcs,
                            recon,
                            residuals_cube,
                            residuals_cube_,
                            frame)
                else:
                    return frame


def cube_collapse(cube, mode='median', n=50, w=None):
    """Collapse a 3D or 4D cube into a 2D frame or 3D cube, respectively.

    The  ``mode`` parameter determines how the collapse should be done. It is
    possible to perform a trimmed mean combination of the frames, as in
    [BRA13]_. In case of a 4D input cube, it is assumed to be an IFS dataset
    with the zero-th axis being the spectral dimension, and the first axis the
    temporal dimension.


    Parameters
    ----------
    cube : numpy ndarray
        Cube.
    mode : {'median', 'mean', 'sum', 'max', 'trimmean', 'absmean', 'wmean'}
        Sets the way of collapsing the images in the cube.
        'wmean' stands for weighted mean and requires weights w to be provided.
        'absmean' stands for the mean of absolute values (potentially useful
        for negfc).
    n : int, optional
        Sets the discarded values at high and low ends. When n = N is the same
        as taking the mean, when n = 1 is like taking the median.
    w: 1d numpy array or list, optional
        Weights to be applied for a weighted mean. Need to be provided if
        collapse mode is 'wmean'.

    Returns
    -------
    frame : numpy ndarray
        Output array, cube combined.
    """
    arr = cube    

    if arr.ndim == 3:
        ax = 0
    elif arr.ndim == 4:
        nch = arr.shape[0]
        ax = 1
    else:
        raise TypeError('The input array is not a cube or 3d array.')

    if mode == 'wmean':
        if w is None:
            raise ValueError(
                "Weights have to be provided for weighted mean mode")
        if len(w) != cube.shape[0]:
            raise TypeError("Weights need same length as cube")
        if isinstance(w, list):
            w = np.array(w)

    if mode == 'mean':
        frame = np.nanmean(arr, axis=ax)
    elif mode == 'median':
        frame = np.nanmedian(arr, axis=ax)
    elif mode == 'sum':
        frame = np.nansum(arr, axis=ax)
    elif mode == 'max':
        frame = np.nanmax(arr, axis=ax)
    elif mode == 'trimmean':
        N = arr.shape[ax]
        k = (N - n)//2
        if N % 2 != n % 2:
            n += 1
        if ax == 0:
            frame = np.empty_like(arr[0])
            for index, _ in np.ndenumerate(arr[0]):
                sort = np.sort(arr[:, index[0], index[1]])
                frame[index] = np.nanmean(sort[k:k+n])
        else:
            frame = np.empty_like(arr[:, 0])
            for j in range(nch):
                for index, _ in np.ndenumerate(arr[:, 0]):
                    sort = np.sort(arr[j, :, index[0], index[1]])
                    frame[j][index] = np.nanmean(sort[k:k+n])
    elif mode == 'wmean':
        #arr[np.where(np.isnan(arr))] = 0  # to avoid product with nan
        if ax == 0:
            frame = np.inner(w, np.moveaxis(arr, 0, -1))
        else:
            frame = np.empty_like(arr[:, 0])
            for j in range(nch):
                frame[j] = np.inner(w, np.moveaxis(arr[j], 0, -1))
    elif mode == 'absmean':
        frame = np.nanmean(np.abs(arr), axis=ax)
    else:
        raise TypeError("mode not recognized")

    return frame

def _project_subtract(
    cube,
    cube_ref,
    ncomp,
    scaling,
    mask_center_px,
    svd_mode,
    verbose,
    full_output,
    indices=None,
    frame=None,
    cube_sig=None,
    left_eigv=False,
    min_frames_pca=10,
):
    """
    PCA projection and model PSF subtraction.

    Used as a helping function by each of the PCA modes (ADI, ADI+RDI,
    ADI+mSDI).

    Parameters
    ----------
    cube : numpy ndarray
        Input cube.
    cube_ref : numpy ndarray
        Reference cube.
    ncomp : int
        Number of principal components.
    scaling : str
        Scaling of pixel values. See ``pca`` docstrings.
    mask_center_px : int
        Masking out a centered circular aperture.
    svd_mode : str
        Mode for SVD computation. See ``pca`` docstrings.
    verbose : bool
        Verbosity.
    full_output : bool
        Whether to return intermediate arrays or not.
    left_eigv : bool, optional
        Whether to use rather left or right singularvectors
    indices : list
        Indices to be used to discard frames (a rotation threshold is used).
    frame : int
        Index of the current frame (when indices is a list and a rotation
        threshold was applied).
    cube_sig: numpy ndarray, opt
        Cube with estimate of significant authentic signals. If provided, this
        will be subtracted from both the cube and the PCA library, before
        projecting the cube onto the principal components.

    Returns
    -------
    ref_lib_shape : int
        [indices is not None, frame is not None] Number of
        rows in the reference library for the given frame.
    residuals: numpy ndarray
        Residuals, returned in every case.
    reconstructed : numpy ndarray
        [full_output=True] The reconstructed array.
    V : numpy ndarray
        [full_output=True, indices is None, frame is None]
        The right singular vectors of the input matrix, as returned by
        ``svd/svd_wrapper()``
    """
    _, y, x = cube.shape

    if not isinstance(ncomp, (int, np.int_, float, np.float16, np.float32,
                              np.float64)):
        raise TypeError("Type not recognized for ncomp, should be int or float")

    # if a cevr is provided instead of an actual ncomp, first calculate it
    if isinstance(ncomp, (float, np.float16, np.float32, np.float64)):
        if not 1 > ncomp > 0:
            raise ValueError(
                "if `ncomp` is float, it must lie in the " "interval (0,1]"
            )

        svdecomp = SVDecomposer(cube, mode="fullfr", svd_mode=svd_mode,
                                scaling=scaling, verbose=verbose)
        _ = svdecomp.get_cevr(plot=False)
        # in this case ncomp is the desired CEVR
        cevr = ncomp
        ncomp = svdecomp.cevr_to_ncomp(cevr)
        if verbose:
            print("Components used : {}".format(ncomp))

    #  if isinstance(ncomp, (int, np.int_)):
    if indices is not None and frame is not None:
        matrix = prepare_matrix(
            cube, scaling, mask_center_px, mode="fullfr", verbose=False
        )
    elif left_eigv:
        matrix = prepare_matrix(cube, scaling, mask_center_px,
                                mode="fullfr", verbose=verbose,
                                discard_mask_pix=True)
    else:
        matrix = prepare_matrix(
            cube, scaling, mask_center_px, mode="fullfr", verbose=verbose
        )
    if cube_sig is None:
        matrix_emp = matrix.copy()
    else:
        if left_eigv:
            matrix_sig = prepare_matrix(cube_sig, scaling, mask_center_px,
                                        mode="fullfr", verbose=verbose,
                                        discard_mask_pix=True)
        else:
            nfr = cube_sig.shape[0]
            matrix_sig = np.reshape(cube_sig, (nfr, -1))
        matrix_emp = matrix - matrix_sig

    if cube_ref is not None:
        if left_eigv:
            matrix_ref = prepare_matrix(cube_sig, scaling, mask_center_px,
                                        mode="fullfr", verbose=verbose,
                                        discard_mask_pix=True)
        else:
            matrix_ref = prepare_matrix(cube_ref, scaling, mask_center_px,
                                        mode="fullfr", verbose=verbose)

    # check whether indices are well defined (i.e. not empty)
    msg = "{} frames comply to delta_rot condition < less than "
    msg1 = msg + "min_frames_pca ({}). Try decreasing delta_rot or "
    msg1 += "min_frames_pca"
    msg2 = msg + "ncomp ({}). Try decreasing the parameter delta_rot or "
    msg2 += "ncomp"
    if indices is not None and frame is not None:
        try:
            ref_lib = matrix_emp[indices]
        except IndexError:
            indices = None
        if cube_ref is None and indices is None:
            raise RuntimeError(msg1.format(0, min_frames_pca))

    # a rotation threshold is used (frames are processed one by one)
    if indices is not None and frame is not None:
        if cube_ref is not None:
            ref_lib = np.concatenate((ref_lib, matrix_ref))
        if ref_lib.shape[0] < min_frames_pca:
            raise RuntimeError(msg1.format(ref_lib.shape[0],
                                           min_frames_pca))
        if ref_lib.shape[0] < ncomp:
            raise RuntimeError(msg2.format(ref_lib.shape[0], ncomp))
        curr_frame = matrix[frame]  # current frame
        curr_frame_emp = matrix_emp[frame]
        if left_eigv:
            V = svd_wrapper(ref_lib, svd_mode, ncomp, False,
                            left_eigv=left_eigv)
            transformed = np.dot(curr_frame_emp.T, V)
            reconstructed = np.dot(V, transformed.T)
        else:
            V = svd_wrapper(ref_lib, svd_mode, ncomp, False)
            transformed = np.dot(curr_frame_emp, V.T)
            reconstructed = np.dot(transformed.T, V)

        residuals = curr_frame - reconstructed

        if full_output:
            return ref_lib.shape[0], residuals, reconstructed
        else:
            return ref_lib.shape[0], residuals

    # the whole matrix is processed at once
    else:
        if cube_ref is not None:
            ref_lib = matrix_ref
        else:
            ref_lib = matrix_emp
        if left_eigv:
            V = svd_wrapper(ref_lib, svd_mode, ncomp, verbose,
                            left_eigv=left_eigv)
            transformed = np.dot(matrix_emp.T, V)
            reconstructed = np.dot(V, transformed.T)
        else:
            V = svd_wrapper(ref_lib, svd_mode, ncomp, verbose)
            transformed = np.dot(V, matrix_emp.T)
            reconstructed = np.dot(transformed.T, V)

        residuals = matrix - reconstructed
        residuals_res = reshape_matrix(residuals, y, x)

        if full_output:
            return residuals_res, reconstructed, V
        else:
            return residuals_res

def pca_annulus(cube, angs, ncomp, annulus_width, r_guess, cube_ref=None,
                svd_mode='lapack', scaling=None, collapse='median',
                weights=None, collapse_ifs='mean', nproc=None, **rot_options):
    """
    PCA-ADI or PCA-RDI processed only for an annulus of the cube, with a given
    width and at a given radial distance to the frame center. It returns a
    processed frame with non-zero values only at the location of the annulus.

    Parameters
    ----------
    cube : 3d or 4d numpy ndarray
        Input data cube to be processed by PCA.
    angs : 1d numpy ndarray
        Vector of derotation angles to align North up in your cube images.
    ncomp : int or list/1d numpy array of int
        The number of principal component.
    annulus_width : float
        The annulus width in pixel on which the PCA is performed.
    r_guess : float
        Radius of the annulus in pixels.
    cube_ref : 3d or 4d numpy ndarray, or list of 3d numpy ndarray, optional
        Reference library cube for Reference Star Differential Imaging.
    svd_mode : {'lapack', 'randsvd', 'eigen', 'arpack'}, str optional
        Switch for different ways of computing the SVD and selected PCs.
    scaling : {None, "temp-mean", spat-mean", "temp-standard",
        "spat-standard"}, None or str optional
        Pixel-wise scaling mode using ``sklearn.preprocessing.scale``
        function. If set to None, the input matrix is left untouched. Otherwise:

        * ``temp-mean``: temporal px-wise mean is subtracted.

        * ``spat-mean``: spatial mean is subtracted.

        * ``temp-standard``: temporal mean centering plus scaling pixel values
          to unit variance (temporally).

        * ``spat-standard``: spatial mean centering plus scaling pixel values
          to unit variance (spatially).

        DISCLAIMER: Using ``temp-mean`` or ``temp-standard`` scaling can improve
        the speckle subtraction for ASDI or (A)RDI reductions. Nonetheless, this
        involves a sort of c-ADI preprocessing, which (i) can be dangerous for
        datasets with low amount of rotation (strong self-subtraction), and (ii)
        should probably be referred to as ARDI (i.e. not RDI stricto sensu).
    collapse : {'median', 'mean', 'sum', 'wmean'}, str or None, optional
        Sets the way of collapsing the residual frames to produce a final image.
        If None then the cube of residuals is returned.
    weights: 1d numpy array or list, optional
        Weights to be applied for a weighted mean. Need to be provided if
        collapse mode is 'wmean' for collapse.
    collapse_ifs : {'median', 'mean', 'sum', 'wmean', 'absmean'}, str or None, optional
        Sets the way of collapsing the spectral frames for producing a final
        image (in the case of a 4D input cube).
    rot_options: dictionary, optional
        Dictionary with optional keyword values for "nproc", "imlib",
        "interpolation, "border_mode", "mask_val",  "edge_blend",
        "interp_zeros", "ker" (see documentation of
        ``vip_hci.preproc.frame_rotate``)

    Returns
    -------
    pca_res: 2d or 3d ndarray
        Depending on ``collapse`` and ``angs`` parameters, either a final
        collapsed frame (``collapse`` not None) or the cube of residuals
        (derotated if angs is not None, non-derotated otherwises).
        Note: for 4d input cube, collapse must be non-None.
    """ 

    def _pca_annulus_3d(cube, angs, ncomp, annulus_width, r_guess, cube_ref,
                        svd_mode, scaling, collapse, weights, **rot_options):

        store_nans = np.isnan(cube)
        cube_nanless = np.copy(cube)
        cube_nanless[store_nans] = 0
        inrad = int(r_guess - annulus_width / 2.)
        outrad = int(r_guess + annulus_width / 2.)
        data, ind = prepare_matrix(cube_nanless, scaling, mode='annular', verbose=False,
                                   inner_radius=inrad, outer_radius=outrad)
        yy, xx = ind

        if cube_ref is not None:
            data_svd, _ = prepare_matrix(cube_ref, scaling, mode='annular',
                                         verbose=False, inner_radius=inrad,
                                         outer_radius=outrad)
        else:
            data_svd = data

        V = svd_wrapper(data_svd, svd_mode, ncomp, verbose=False)

        transformed = np.dot(data, V.T)
        reconstructed = np.dot(transformed, V)
        residuals = data - reconstructed
        cube_zeros = np.zeros_like(cube)
        cube_zeros[:, yy, xx] = residuals

        cube_zeros[store_nans] = np.nan

        if angs is not None:
                        # Cube is derotated according to the parallactic angle and collapsed
            origin = np.asarray([np.shape(cube_zeros)[1] / 2 - 0.5, np.shape(cube_zeros)[2] / 2 - 0.5])
            cube_res_der = rotate_imgs(cube_zeros, angs, np.array(len(cube_zeros)*[origin]), numthreads=nproc, flipx=False, new_center=origin)
            #cube_res_der = cube_derotate(cube_zeros, angs, **rot_options)
            if collapse is not None:
                pca_frame = cube_collapse(cube_res_der, mode=collapse,
                                          w=weights)
                return pca_frame
            else:
                return cube_res_der

        else:
            if collapse is not None:
                pca_frame = cube_collapse(cube_zeros, mode=collapse, w=weights)
                return pca_frame
            else:
                return cube_zeros

    if cube.ndim == 3:
        return _pca_annulus_3d(cube, angs, ncomp, annulus_width, r_guess,
                               cube_ref, svd_mode, scaling, collapse, weights,
                               **rot_options)
    elif cube.ndim == 4:
        nch = cube.shape[0]
        if cube_ref is not None:
            if cube_ref.ndim == 3:
                cube_ref = [cube_ref]*nch
        if np.isscalar(ncomp):
            ncomp = [ncomp]*nch
        elif isinstance(ncomp, list) and len(ncomp) != nch:
            msg = "If ncomp is a list, in the case of a 4d input cube without "
            msg += "input scale_list, it should have the same length as the "
            msg += "first dimension of the cube."
            raise TypeError(msg)
        if collapse is None:
            raise ValueError("mode not supported. Provide value for collapse")
        ifs_res = np.zeros([nch, cube.shape[2], cube.shape[3]])
        for ch in range(nch):
            if cube_ref is not None:
                if cube_ref[ch].ndim != 3:
                    msg = "Ref cube has wrong format for 4d input cube"
                    raise TypeError(msg)
                cube_ref_tmp = cube_ref[ch]
            else:
                cube_ref_tmp = cube_ref
            ifs_res[ch] = _pca_annulus_3d(cube[ch], angs, ncomp[ch],
                                          annulus_width, r_guess, cube_ref_tmp,
                                          svd_mode, scaling, collapse, weights,
                                          **rot_options)
        return cube_collapse(ifs_res, mode=collapse_ifs)

'''
def frame_rotate(array, angle, imlib='vip-fft', interpolation='lanczos4',
                 cxy=None, border_mode='constant', mask_val=np.nan,
                 edge_blend=None, interp_zeros=False, ker=1):
    """Rotate a frame or 2D array.

    Parameters
    ----------
    array : numpy ndarray
        Input image, 2d array.
    angle : float
        Rotation angle.
    imlib : {'opencv', 'skimage', 'vip-fft', 'torch-fft'}, str optional
        Library used for image transformations. Opencv is faster than skimage or
        'vip-fft', but vip-fft slightly better preserves the flux in the image
        (followed by skimage with a biquintic interpolation). 'vip-fft'
        corresponds to the FFT-based rotation method described in [LAR97]_, and
        implemented in this module. Best results are obtained with images
        without any sharp intensity change (i.e. no numerical mask).
        Edge-blending and/or zero-interpolation may help if sharp transitions
        are unavoidable.
    interpolation : str, optional
        [Only used for imlib='opencv' or imlib='skimage']
        For Skimage the options are: 'nearneig', bilinear', 'biquadratic',
        'bicubic', 'biquartic' or 'biquintic'. The 'nearneig' interpolation is
        the fastest and the 'biquintic' the slowest. The 'nearneig' is the
        poorer option for interpolation of noisy astronomical images.
        For Opencv the options are: 'nearneig', 'bilinear', 'bicubic' or
        'lanczos4'. The 'nearneig' interpolation is the fastest and the
        'lanczos4' the slowest and more accurate. 'lanczos4' is the default for
        Opencv and 'biquartic' for Skimage.
    cxy : float, optional
        Coordinates X,Y  of the point with respect to which the rotation will be
        performed. By default the rotation is done with respect to the center
        of the frame.
    border_mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, str opt
        Pixel extrapolation method for handling the borders. 'constant' for
        padding with zeros. 'edge' for padding with the edge values of the
        image. 'symmetric' for padding with the reflection of the vector
        mirrored along the edge of the array. 'reflect' for padding with the
        reflection of the vector mirrored on the first and last values of the
        vector along each axis. 'wrap' for padding with the wrap of the vector
        along the axis (the first values are used to pad the end and the end
        values are used to pad the beginning). Default is 'constant'.
    mask_val: flt, opt
        If any numerical mask in the image to be rotated, what are its values?
        Will only be used if a strategy to mitigate Gibbs effects is adopted -
        see below.
    edge_blend: str, opt {None,'noise','interp','noise+interp'}
        Whether to blend the edges, by padding nans then inter/extrapolate them
        with a gaussian filter. Slower but can significantly reduce ringing
        artefacts from Gibbs phenomenon, in particular if several consecutive
        rotations are involved in your image processing.

        - 'noise': pad with small amplitude noise inferred from neighbours
        - 'interp': interpolated from neighbouring pixels using Gaussian kernel.
        - 'noise+interp': sum both components above at masked locations.

        Original mask will be placed back after rotation.
    interp_zeros: bool, opt
        [only used if edge_blend is not None]
        Whether to interpolate zeros in the frame before (de)rotation. Not
        dealing with them can induce a Gibbs phenomenon near their location.
        However, this flag should be false if rotating a binary mask.
    ker: float, opt
        Size of the Gaussian kernel used for interpolation.

    Returns
    -------
    array_out : numpy ndarray
        Resulting frame.

    """
    if array.ndim != 2:
        raise TypeError('Input array is not a frame or 2d array')

    if edge_blend is None:
        edge_blend = ''

    if edge_blend != '' or imlib in ['vip-fft', 'torch-fft']:
        # fill with nans
        cy_ori, cx_ori = frame_center(array)
        y_ori, x_ori = array.shape
        if np.isnan(mask_val):
            mask_ori = np.where(np.isnan(array))
        else:
            mask_ori = np.where(array == mask_val)
        array_nan = array.copy()
        array_zeros = array.copy()
        if interp_zeros == 1 or mask_val != 0:  # set to nans for interpolation
            array_nan[np.where(array == mask_val)] = np.nan
        else:
            array_zeros[np.where(np.isnan(array))] = 0
        if 'noise' in edge_blend:
            # evaluate std and med far from the star, avoiding nans
            _, med, stddev = sigma_clipped_stats(array_nan, sigma=1.5,
                                                 cenfunc=np.nanmedian,
                                                 stdfunc=np.nanstd)

        # pad and interpolate, about 1.2x original size
        if imlib in ['vip-fft', 'torch-fft']:
            fac = 1.5
        else:
            fac = 1.1
        new_y = int(y_ori*fac)
        new_x = int(x_ori*fac)
        if y_ori % 2 != new_y % 2:
            new_y += 1
        if x_ori % 2 != new_x % 2:
            new_x += 1
        array_prep = np.empty([new_y, new_x])
        array_prep1 = np.zeros([new_y, new_x])
        array_prep[:] = np.nan
        if 'interp' in edge_blend:
            array_prep2 = array_prep.copy()
            med = 0  # local level will be added with Gaussian kernel
        if 'noise' in edge_blend:
            array_prep = np.random.normal(loc=med, scale=stddev,
                                          size=(new_y, new_x))
        cy, cx = frame_center(array_prep)
        y0_p = int(cy-cy_ori)
        y1_p = int(cy+cy_ori)
        if new_y % 2:
            y1_p += 1
        x0_p = int(cx-cx_ori)
        x1_p = int(cx+cx_ori)
        if new_x % 2:
            x1_p += 1
        if interp_zeros:
            array_prep[y0_p:y1_p, x0_p:x1_p] = array_nan.copy()
            array_prep1[y0_p:y1_p, x0_p:x1_p] = array_nan.copy()
        else:
            array_prep[y0_p:y1_p, x0_p:x1_p] = array_zeros.copy()
        # interpolate nans with a Gaussian filter
        if 'interp' in edge_blend:
            array_prep2[y0_p:y1_p, x0_p:x1_p] = array_nan.copy()
            cond1 = array_prep1 == 0
            cond2 = np.isnan(array_prep2)
            new_nan = np.where(cond1 & cond2)
            mask_nan = np.where(np.isnan(array_prep2))
            if not ker:
                ker = array_nan.shape[0]/5
            ker2 = 1
            array_prep_corr1 = frame_filter_lowpass(array_prep2, mode='gauss',
                                                    fwhm_size=ker)
            if 'noise' in edge_blend:
                array_prep_corr2 = frame_filter_lowpass(array_prep2,
                                                        mode='gauss',
                                                        fwhm_size=ker2)
                ori_nan = np.where(np.isnan(array_prep1))
                array_prep[ori_nan] = array_prep_corr2[ori_nan]
                array_prep[new_nan] += array_prep_corr1[new_nan]
            else:
                array_prep[mask_nan] = array_prep_corr1[mask_nan]

        # finally pad zeros for 4x larger images before FFT
        if imlib == 'vip-fft':
            array_prep, new_idx = frame_pad(array_prep, fac=4/fac, fillwith=0,
                                            full_output=True)
            y0 = new_idx[0]+y0_p
            y1 = new_idx[0]+y1_p
            x0 = new_idx[2]+x0_p
            x1 = new_idx[2]+x1_p
        else:
            y0 = y0_p
            y1 = y1_p
            x0 = x0_p
            x1 = x1_p
    else:
        array_prep = array.copy()

    # residual (non-interp) nans should be set to 0 to avoid bug in rotation
    array_prep[np.where(np.isnan(array_prep))] = 0

    y, x = array_prep.shape

    if cxy is None:
        cy, cx = frame_center(array_prep)
    else:
        cx, cy = cxy
        cond_imlib = imlib in ['vip-fft', 'torch-fft']
        if cond_imlib and (cy, cx) != frame_center(array_prep):
            msg = "'vip-fft' imlib does not yet allow for custom center to be "
            msg += " provided "
            raise ValueError(msg)

    if imlib == 'vip-fft':
        array_out = rotate_fft(array_prep, angle)

    elif imlib == 'skimage':
        if interpolation == 'nearneig':
            order = 0
        elif interpolation == 'bilinear':
            order = 1
        elif interpolation == 'biquadratic':
            order = 2
        elif interpolation == 'bicubic':
            order = 3
        elif interpolation == 'biquartic' or interpolation == 'lanczos4':
            order = 4
        elif interpolation == 'biquintic':
            order = 5
        else:
            raise ValueError('Skimage interpolation method not recognized')

        if border_mode not in ['constant', 'edge', 'symmetric', 'reflect',
                               'wrap']:
            raise ValueError('Skimage `border_mode` not recognized.')

        # for a non-constant image, normalize manually
        min_val = np.nanmin(array_prep)
        max_val = np.nanmax(array_prep)
        if min_val != max_val:
            norm = True
            im_temp = array_prep - min_val
            max_val = np.nanmax(im_temp)
            im_temp /= max_val
        else:
            norm = False
            im_temp = array_prep.copy()

        array_out = rotate(im_temp, angle, order=order, center=(cx, cy),
                           cval=0, mode=border_mode)

        if norm:
            array_out *= max_val
            array_out += min_val
        array_out = np.nan_to_num(array_out, copy=False)

    elif imlib == 'opencv':
        if no_opencv:
            msg = 'Opencv python bindings cannot be imported. Install opencv or'
            msg += ' set imlib to skimage'
            raise RuntimeError(msg)

        if interpolation == 'bilinear':
            intp = cv2.INTER_LINEAR
        elif interpolation == 'bicubic':
            intp = cv2.INTER_CUBIC
        elif interpolation == 'nearneig':
            intp = cv2.INTER_NEAREST
        elif interpolation == 'lanczos4':
            intp = cv2.INTER_LANCZOS4
        else:
            raise ValueError(f'Opencv interpolation method `{interpolation}` is not recognized')

        if border_mode == 'constant':
            bormo = cv2.BORDER_CONSTANT  # iiiiii|abcdefgh|iiiiiii
        elif border_mode == 'edge':
            bormo = cv2.BORDER_REPLICATE  # aaaaaa|abcdefgh|hhhhhhh
        elif border_mode == 'symmetric':
            bormo = cv2.BORDER_REFLECT  # fedcba|abcdefgh|hgfedcb
        elif border_mode == 'reflect':
            bormo = cv2.BORDER_REFLECT_101  # gfedcb|abcdefgh|gfedcba
        elif border_mode == 'wrap':
            bormo = cv2.BORDER_WRAP  # cdefgh|abcdefgh|abcdefg
        else:
            raise ValueError('Opencv `border_mode` not recognized.')

        M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
        array_out = cv2.warpAffine(array_prep.astype(np.float32), M, (x, y),
                                   flags=intp, borderMode=bormo)
    elif imlib == 'torch-fft':
        if no_torch:
            msg = 'Pytorch bindings cannot be imported. Install torch or'
            msg += ' set imlib to skimage'
            raise RuntimeError(msg)

        array_out = (tensor_rotate_fft(torch.unsqueeze(
            torch.from_numpy(array_prep), 0), angle)[0]).numpy()

    else:
        raise ValueError('Image transformation library not recognized')

    if edge_blend != '' or imlib in ['vip-fft', 'torch-fft']:
        array_out = array_out[y0:y1, x0:x1]  # remove padding
        array_out[mask_ori] = mask_val      # mask again original masked values

    return array_out


def cube_derotate(array, angle_list, imlib='vip-fft', interpolation='lanczos4',
                  cxy=None, nproc=1, border_mode='constant', mask_val=np.nan,
                  edge_blend=None, interp_zeros=False, ker=1):
    """Rotate a cube (3d array or image sequence) providing a vector or\
    corresponding angles.

    Serves for rotating an ADI sequence to a common north given a vector with
    the corresponding parallactic angles for each frame.

    Parameters
    ----------
    array : numpy.ndarray
        Input 3d array, cube.
    angle_list : list or 1D numpy.ndarray
        Vector containing the parallactic angles.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    cxy : tuple of int, optional
        Coordinates X,Y  of the point with respect to which the rotation will be
        performed. By default the rotation is done with respect to the center
        of the frames, as it is returned by the function
        vip_hci.var.frame_center.
    nproc : int, optional
        Whether to rotate the frames in the sequence in a multi-processing
        fashion. Only useful if the cube is significantly large (frame size and
        number of frames).
    border_mode : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    mask_val: flt, opt
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    edge_blend : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interp_zeros : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    ker: int, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.

    Returns
    -------
    array_der : numpy ndarray
        Resulting cube with de-rotated frames.

    """
    if array.ndim != 3:
        raise TypeError('Input array is not a cube or 3d array.')
    n_frames = array.shape[0]

    if nproc is None:
        nproc = cpu_count() // 2        # Hyper-threading doubles the # of cores

    if nproc == 1:
        array_der = np.zeros_like(array)
        for i in range(n_frames):
            array_der[i] = frame_rotate(array[i], -angle_list[i], imlib=imlib,
                                        interpolation=interpolation, cxy=cxy,
                                        border_mode=border_mode,
                                        mask_val=mask_val,
                                        edge_blend=edge_blend,
                                        interp_zeros=interp_zeros, ker=ker)
    elif nproc > 1:

        res = pool_map(nproc, _frame_rotate_mp, iterable(array),
                       iterable(-angle_list), imlib, interpolation, cxy,
                       border_mode, mask_val, edge_blend, interp_zeros, ker)
        array_der = np.array(res)

    return array_der
'''