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
from pyklip.klip import rotate as klip_rotate
from vip_hci.preproc.derotation import _find_indices_adi, _define_annuli
from vip_hci.stats import descriptive_stats
from vip_hci.var import get_annulus_segments, matrix_scaling, prepare_matrix, reshape_matrix
from vip_hci.config.paramenum import SvdMode, Imlib, Interpolation, Collapse, ALGO_KEY
import matplotlib.pyplot as plt

AUTO = "auto"

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
            