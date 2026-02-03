#----------------------------------------
# IMPORTS
#----------------------------------------
import numpy as np
from itertools import groupby
import os, pathlib, psutil
import matplotlib.pyplot as plt
from astropy.io import fits
#from astropy.time import Time
from tqdm.auto import tqdm
from astropy.convolution import convolve, convolve_fft,\
    Box2DKernel, Gaussian1DKernel, Ring2DKernel
from scipy.stats import linregress
from scipy.signal import correlate
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
from image_registration import chi2_shift
from multiprocessing.pool import ThreadPool as Pool
from LBT_vip_pca import pca_annular, pca_annulus

#----------------------------------------
# CLASSES
#----------------------------------------
class IntegrateFrames(object):

        def __init__(self, params):
            self.params = params
        
        def __call__(self, files):

                array_shape = self.params

                # array for integrating files      
                buf_3D = np.zeros((len(files), array_shape[0], array_shape[1]))

                count = 0

                for k in range(len(files)):

                        #try:
                        hdul = fits.open(files[k])
                    
                        buf_3D[k] = hdul[0].data

                        hdul.close()

                        count += 1
                                
                       # except:
                       #         buf_3D[k][:] = np.nan 
            
                return np.nanmean(buf_3D, axis=0)*count, count

class BinFrames(object):

        def __init__(self, params):
            self.params = params
        
        def __call__(self, angles_files_tuple):

                array_shape, binned_dir = self.params

                angles, files = angles_files_tuple

                # array for integrating files      
                buf_3D = np.zeros((len(files), array_shape[0], array_shape[1]))

                for k in range(len(files)):

                        hdul = fits.open(files[k])
                    
                        buf_3D[k] = hdul[0].data

                        hdul.close()

                filename = files[0].name

                binned_frame = np.nanmean(buf_3D, axis=0)
                newhdul = fits.HDUList([fits.PrimaryHDU(data=binned_frame)])
                newhdul.writeto(os.path.join(binned_dir, "binned_"+filename), overwrite=True)
                newhdul.close()
        
                return np.nanmean(angles), "binned_"+filename

class BinFramesMem(object):

        def __init__(self, params):
            self.params = params
        
        def __call__(self, angles_files_tuple):

                array_shape = self.params

                angles, files = angles_files_tuple

                # array for integrating files      
                buf_3D = np.zeros((len(files), array_shape[0], array_shape[1]))

                for k in range(len(files)):

                        hdul = fits.open(files[k])
                    
                        buf_3D[k] = hdul[0].data

                        hdul.close()

                filename = files[0].name

                binned_frame = np.nanmean(buf_3D, axis=0)
        
                return binned_frame, np.nanmean(angles), "binned_"+filename

class FileInfoHighPass(object):

    def __init__(self, params):
        self.params = params
        
    def __call__(self, file):

        highpass_dir, bools, obj, skip_target_check = self.params

        hdul = fits.open(file)

        image = hdul[0].data[0]
        
        # Check if correct object
        if not skip_target_check:
            try:
                object = hdul[0].header['OBJNAME']
            except:
                print(file)
                raise ValueError("OBJNAME header not found")
            if object != obj:
                raise ValueError("Object in header does not match given object")
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
        try:
            chop = hdul[0].header['CHOP_POS']
        except:
            chop = "CHOP_NA"

        para_angle = hdul[0].header['LBT_PARA']

        frame_median = np.median(hdul[0].data[0])

        new_image = np.copy(image)
        new_image[~bools] = np.median(image)
        new_image = np.pad(new_image, 20, mode='linear_ramp')

        filtered_frame = image - convolve_fft(new_image, Box2DKernel(30))[20:-20, 20:-20]
        filtered_frame[~bools] = np.nan
        newhdul = fits.HDUList([fits.PrimaryHDU(data=filtered_frame)])
        newhdul.writeto(os.path.join(highpass_dir, "highpass_"+file.name), overwrite=True)
        hdul.close()
        newhdul.close()
    
        return chop, frame_median, para_angle

class SubtractBackground(object):

    def __init__(self, params):
        
        self.params = params
        
    def __call__(self, i):

        subtracted_dir, files, framew, frameh, edge_cut, channel_edges, vertical_lines, flat = self.params

        if i == 0:
            
            hdul = fits.open(files[1])
            bg = np.float64(hdul[0].data[0])
    
        elif i == (len(files) -1):
            
            hdul = fits.open(files[len(files)-2])
            bg = np.float64(hdul[0].data[0]) 
    
        else:
            
            hdul = fits.open(files[i-1])
            hdul2 = fits.open(files[i+1])
            bg = 0.5*(hdul[0].data[0] + hdul2[0].data[0])
            hdul2.close()
            
        hdul.close()
    
        unsubtracted = fits.open(files[i])
    
        subtracted_frame = (unsubtracted[0].data[0] - bg)/flat

        for channel_edge in channel_edges:
            subtracted_frame = repairChannelEdges(subtracted_frame, channel_edge)
        for vertical_line in vertical_lines:
            subtracted_frame = repairVerticalLine(subtracted_frame, vertical_line)
    
        offsets = np.mean(subtracted_frame[:,int(framew/2 - 6):int(framew/2 - 1)] - subtracted_frame[:,int(framew/2):int(framew/2 + 5)], axis=1)
        xoff = np.linspace(0, len(offsets)-1, len(offsets))
        p = np.polyfit(xoff, offsets, deg=3)
        subtracted_frame[:, int(framew/2):] =  subtracted_frame[:, int(framew/2):] +\
                np.asarray([p[0]*xoff**3 + p[1]*xoff**2 + p[2]*xoff + p[3]]*int(framew/2)).T
        subtracted_frame = subtracted_frame[edge_cut:-1*edge_cut ,edge_cut :-1*edge_cut ]
        
        framew -= 2*edge_cut
        frameh -= 2*edge_cut
        
        max_indices = np.where(subtracted_frame == np.nanmax(subtracted_frame))
        min_indices = np.where(subtracted_frame == np.nanmin(subtracted_frame))
        
        maximum = (max_indices[1][0], max_indices[0][0])

        minimum = (min_indices[1][0], min_indices[0][0])
        
        max_mask =  circular_mask((maximum[0], maximum[1]), 35, framew, framew)
        min_mask = circular_mask((minimum[0], minimum[1]), 35, framew, framew)
        max_aperture =  circular_mask((maximum[0], maximum[1]), 40, framew, framew) ^ max_mask
        min_aperture = circular_mask((minimum[0], minimum[1]), 40, framew, framew) ^ min_mask
        
        new_bg = np.copy(subtracted_frame)
        new_bg[max_mask] = np.median(new_bg[max_aperture])
        new_bg[min_mask] = np.median(new_bg[min_aperture])
        subtracted_frame = subtracted_frame - convolve_fft(np.pad(new_bg, 50, mode='edge'), Ring2DKernel(25, 20))[50:-50, 50:-50]
    
        newhdul = fits.HDUList([fits.PrimaryHDU(data=(subtracted_frame))])
        newhdul.writeto(os.path.join(subtracted_dir, "subtracted_"+files[i].name), overwrite=True)
        newhdul.close()
    
        unsubtracted.close()

        return minimum, maximum

class RegisterFrames(object):

    def __init__(self, params):
        
        self.params = params
        
    def __call__(self, file):

        subtracted_dir, aligned_dir, padding_x, padding_y, center_padding_x, center_padding_y,\
        px, py, reference, first_maxima, windowsize, nan_mask_diameter = self.params
        
        hdul = fits.open(os.path.join(subtracted_dir, "subtracted_"+file.name))
        frame = hdul[0].data
        hdul.close()

        maxima = np.where(frame == np.nanmax(frame))
        minima = np.where(frame == np.nanmin(frame))
    
        cutout = frame[(maxima[0][0]-windowsize):(maxima[0][0]+windowsize),(maxima[1][0]-windowsize):(maxima[1][0]+windowsize)]
        try:
            offset_x, offset_y = chi2_shift(reference, cutout, upsample_factor='auto', return_error=False)
        except:
            offset_x = 0
            offset_y = 0
            
        offset_x += (maxima[1][0] - first_maxima[1][0])
        offset_y += (maxima[0][0] - first_maxima[0][0])

        frame[circular_mask((minima[1][0], minima[0][0]), nan_mask_diameter, len(px)-np.abs(padding_x), len(py)-np.abs(padding_y))] = np.nan
        frame = align_frame(frame, px, py, padding_x, padding_y, offset_x, offset_y)
        frame = pad_frame(frame, len(px) + np.abs(center_padding_x), len(py) + np.abs(center_padding_y), center_padding_x, center_padding_y)

  
        newhdul = fits.HDUList([fits.PrimaryHDU(data=(frame))])
        newhdul.writeto(os.path.join(aligned_dir, "aligned_"+file.name), overwrite=True)
        newhdul.close()
        
        psfmaxima = np.nanmax(frame)
        background_dev = np.nanstd(frame[np.abs(frame - np.nanmedian(frame)) < 3*np.nanstd(frame)])

        return psfmaxima, background_dev

class EvaluateFrames(object):
    
    def __init__(self, params):
        
        self.params = params
        
    def __call__(self, chop_file_tuple):
        
        chopa_mean_frame, chopb_mean_frame, wx, wy, windowsize, array_shape = self.params

        chop, file = chop_file_tuple
        
        hdul = fits.open(file)
        frame = hdul[0].data
        frame[np.isnan(frame)] = 0
        hdul.close()

        psfmaxima = np.nanmax(frame)
        background_dev = np.nanstd(frame[np.abs(frame - np.nanmedian(frame)) < 3*np.nanstd(frame)])
        
        if chop == "CHOP_B":
            corr = (np.max(correlate(chopb_mean_frame, frame)))
        else:
            corr = (np.max(correlate(chopa_mean_frame, frame)))    
    
        cutout = frame[(int(array_shape[0]/2)-windowsize):(int(array_shape[0]/2)+windowsize),(int(array_shape[1]/2)-windowsize):(int(array_shape[1]/2)+windowsize)]
        
        try:
            reffit, _ = curve_fit(Gaussian2D_ravel, (wx, wy), cutout.ravel(), p0=[np.max(cutout),\
                                10, 10, -1, windowsize, windowsize],\
                               bounds=([0, 1, 1, 1*-np.inf, 1, 1], [2*np.max(cutout), 200, 200, np.inf, 2*windowsize, 2*windowsize]))
        except:
            
            return psfmaxima, background_dev, corr, np.nan, np.nan, np.nan, np.nan
            
        return psfmaxima, background_dev, corr, reffit[0], reffit[1], reffit[2], reffit[3]

class EvaluateFramesMem(object):
    
    def __init__(self, params):
        
        self.params = params
        
    def __call__(self, chop_file_tuple):
        
        img_files, chopa_mean_frame, chopb_mean_frame, wx, wy, windowsize, array_shape = self.params

        chop, index = chop_file_tuple
        frame = img_files[int(index)]

        psfmaxima = np.nanmax(frame)
        background_dev = np.nanstd(frame[np.abs(frame - np.nanmedian(frame)) < 3*np.nanstd(frame)])
        
        if chop == "CHOP_B":
            corr = (np.max(correlate(chopb_mean_frame, frame)))
        else:
            corr = (np.max(correlate(chopa_mean_frame, frame)))    
    
        cutout = frame[(int(array_shape[0]/2)-windowsize):(int(array_shape[0]/2)+windowsize),(int(array_shape[1]/2)-windowsize):(int(array_shape[1]/2)+windowsize)]
        
        try:
            reffit, _ = curve_fit(Gaussian2D_ravel, (wx, wy), cutout.ravel(), p0=[np.max(cutout),\
                                10, 10, -1, windowsize, windowsize],\
                               bounds=([0, 1, 1, 1*-np.inf, 1, 1], [2*np.max(cutout), 200, 200, np.inf, 2*windowsize, 2*windowsize]))
        except:
            
            return psfmaxima, background_dev, corr, np.nan, np.nan, np.nan, np.nan
            
        return psfmaxima, background_dev, corr, reffit[0], reffit[1], reffit[2], reffit[3]

class InjectPlanet(object):

    def __init__(self, params):
        self.params = params
    
    def __call__(self, info):

            wx, wy, directory, array_shape, radius, theta, ratio, width, height, highpassrad = self.params

            planet = Gaussian2D((wx, wy), ratio*info[1],info[2],info[3], 0,\
                 0.5*(array_shape[0] - 1) + radius*np.cos((theta - info[4])*np.pi/180), 0.5*(array_shape[1] - 1) + radius*np.sin((theta - info[4])*np.pi/180))
        
            hdul = fits.open(info[0])
            img = hdul[0].data[width[0]:width[1], height[0]:height[1]] + planet
            hdul.close()

            if highpassrad != None:

                max_indices = np.where(img == np.nanmax(img))
                maximum = (max_indices[1][0], max_indices[0][0])

                fwhm = 2*np.sqrt(2*np.log(2)*(info[2]+info[3]))
                
                max_mask =  circular_mask((maximum[0], maximum[1]), 1.1*fwhm, array_shape[1], array_shape[0])
                max_aperture =  circular_mask((maximum[0], maximum[1]), 1.1*1.1*fwhm, array_shape[1], array_shape[0]) ^ max_mask
                
                new_bg = np.copy(img)
                new_bg[max_mask] = np.median(new_bg[max_aperture])
                
                img = img - convolve_fft(np.pad(new_bg, 50, mode='edge'), Ring2DKernel(int(highpassrad*5/4), highpassrad))[50:-50, 50:-50]
                
            newhdul = fits.HDUList([fits.PrimaryHDU(data=(img))])      

            newhdul.writeto(os.path.join(directory, "injected_"+info[0].name), overwrite=True)
            newhdul.close()

            return True, True

class InjectPlanetMem(object):

    def __init__(self, params):
        self.params = params
    
    def __call__(self, info):

            imgs, wx, wy, array_shape, radius, theta, ratio, width, height, highpassrad = self.params
            '''
            hdul = fits.open(info[0])
            img = hdul[0].data[width[0]:width[1], height[0]:height[1]]
            hdul.close()
            '''

            planet = Gaussian2D((wx, wy), ratio*info[1],info[2],info[3], 0,\
                 0.5*(array_shape[0] - 1) + radius*np.cos((theta - info[4])*np.pi/180), 0.5*(array_shape[1] - 1) + radius*np.sin((theta - info[4])*np.pi/180))
        
            img = imgs[int(info[0])][width[0]:width[1], height[0]:height[1]] + planet

            if highpassrad != None:
                try:
                    max_indices = np.where(img == np.nanmax(img))
                    
                    maximum = (max_indices[1][0], max_indices[0][0])
                except:
                    plt.imshow(img)
                    plt.show()
                    raise ValueError("nanmax failed")
                fwhm = 2*np.sqrt(2*np.log(2)*(info[2]+info[3]))
                
                max_mask =  circular_mask((maximum[0], maximum[1]), 1.1*fwhm, array_shape[1], array_shape[0])
                max_aperture =  circular_mask((maximum[0], maximum[1]), 1.1*1.1*fwhm, array_shape[1], array_shape[0]) ^ max_mask
                
                new_bg = np.copy(img)
                new_bg[max_mask] = np.median(new_bg[max_aperture])
                
                img = img - convolve_fft(np.pad(new_bg, 50, mode='edge'), Ring2DKernel(int(highpassrad*5/4), highpassrad))[50:-50, 50:-50]
                
            return img, info[0]


class GetContrasts(object):

        def __init__(self, params):
            self.params = params
        
        def __call__(self, coords):

                binned_files, binned_amps, binned_sigmax, binned_sigmay, binned_angles, curve_prior, array_shape,\
                         hwhm, file_save, tolerance, high_buffer, delta_rot, n_segments, max_iterations, highpassrad, ncomp, add_info = self.params                    
    
                last_contrast = high_buffer*curve_prior(coords[1])
    
                snr_plot = np.array([])
                contrast_plot = np.array([])
                            
                anticorrelation = False
                correlation = False
                lin_error = False
                counter = 0
            
                while True:
                    #try:
                        
                        curr_snr, curr_noise_factor =\
                            forward_model(binned_files, binned_amps, binned_sigmax, binned_sigmay, binned_angles, array_shape,\
                                          hwhm, injection_radius=coords[1], injection_angle=coords[0], initial_contrast=last_contrast,\
                                          crop_size=2*int(np.ceil(1.05*(coords[1]+hwhm))), delta_rot=delta_rot, n_segments=n_segments, file_save=file_save, ncomp=ncomp, highpassrad=highpassrad, nproc=1)
                        snr_plot = np.append(snr_plot, curr_snr)
                        contrast_plot = np.append(contrast_plot, last_contrast)
                        #print("Contrast: ",last_contrast)
                        #print("SNR: ",curr_snr)
        
                        if (np.abs(curr_snr - 5)/(5) < tolerance):
                            
                            break
                            
                        else:
                            
                            if np.max(snr_plot) > 5:
                                
                                if (len(snr_plot) > 1):
                                    
                                    results = linregress(snr_plot[-2:], np.log(contrast_plot[-2:]))
        
                                    curr_approx_contrast = np.exp(np.round(results[0]*5 + results[1], 3)) 
                                    #print("Linregress: ", curr_approx_contrast)
                                
                                    if curr_approx_contrast < 1e-6 or curr_approx_contrast> 1e-1:
                                        
                                        if curr_snr < 0:
                                            best_snr = snr_plot[1:][np.nanargmin(np.abs(snr_plot[1:] - 5))]
                                            best_contrast = contrast_plot[1:][np.nanargmin(np.abs(snr_plot[1:] - 5))]
                                            snr_plot = snr_plot[:-1]
                                            contrast_plot = contrast_plot[:-1]
                                        else:
                                            best_snr = curr_snr
                                            best_contrast = last_contrast
                                            
                                        curr_approx_contrast = np.abs((5/best_snr)*(best_contrast))
                                        if np.isin(curr_approx_contrast, contrast_plot):                                        
                                            results = linregress(snr_plot[snr_plot > 0], np.log(contrast_plot[snr_plot > 0]))
                                            curr_approx_contrast = np.exp(np.round(results[0]*5 + results[1], 3)) 
    
                                else:
                                    
                                    curr_approx_contrast = np.abs((5/curr_snr)*(last_contrast))
                            else:
                                if len(snr_plot) > 1:
                                    if ((snr_plot[-1] < snr_plot[-2]) & (contrast_plot[-1] > contrast_plot[-2])) or\
                                            ((snr_plot[-1] > snr_plot[-2]) & (contrast_plot[-1] < contrast_plot[-2])):
                                        anticorrelation = True
                                        curr_approx_contrast = 0.25*last_contrast
                                    if ((snr_plot[-1] > snr_plot[-2]) & (contrast_plot[-1] > contrast_plot[-2])) or\
                                            ((snr_plot[-1] < snr_plot[-2]) & (contrast_plot[-1] < contrast_plot[-2])):
                                        correlation = True
                                        curr_approx_contrast = 2*last_contrast
                                else:
                                    curr_approx_contrast = 2*last_contrast
                                    
                            last_contrast = curr_approx_contrast
        
                        counter += 1

        
                        if anticorrelation & correlation or counter > max_iterations or lin_error:
                            
                            try:
                                
                                last_contrast = contrast_plot[np.nanargmin(np.abs(snr_plot - 5))]
                                curr_snr = snr_plot[np.nanargmin(np.abs(snr_plot - 5))]
                                
                            except:
                                
                                last_contrast = np.nan
                                curr_snr = np.nan
                                
                            if anticorrelation & correlation:
                                print("Absolute max reached: ", last_contrast)
                            else:
                                print("Max iterations reached: ", last_contrast)
                            break

                return last_contrast, curr_snr
'''
                    except:
                        
                        lin_error=True
                        np.savez("Error_log-"+"_radius-"+str(coords[1])+"_angle-"+str(coords[0])+"_binning-"+str(add_info[0])+"_sigma-"+str(add_info[1])+"_ncomp-"+str(ncomp)+"_highpass-"+\
                                 str(highpassrad)+".npz", coords, add_info[0], add_info[1], ncomp, highpassrad)
                        pass
'''
#----------------------------------------
# FUNCTIONS
#----------------------------------------

def combine(obj, raw_dir, master_badmap_dir, combn, side, testing=False, test_number=None, start_frame=None,\
            end_frame = None, skip_target_check=False,background_limit = 28000, threadcount=50):

    master_badmap = (fits.open(master_badmap_dir))[0].data
    master_badmap[:,:3] = 0
    master_badmap[:,-3:] = 0
    master_badmap[:3,:] = 0
    master_badmap[-3:,:] = 0
    
    bools = np.full(np.shape(master_badmap), False)
    bools[master_badmap > 0] = True
    
    if combn > 0:
        skip_target_check = True

    root_dir = os.path.dirname(raw_dir)
    
    highpass_dir=os.path.join(root_dir,'highpass')

    if not os.path.exists(highpass_dir):
        os.makedirs(highpass_dir)
    
    if combn == 0:
        search_dir=raw_dir
    if combn == 1:
        search_dir=root_dir+'skys1/'
    if combn == 2:
        search_dir=root_dir+'skys2/'
    if combn == 3:
        search_dir=root_dir+'skys3/'

    print('Searching in ',search_dir)
    
    # directory for all files
    files = sorted(list(pathlib.Path(str(search_dir)).rglob('*.fits')))
    print("Detected ", len(files), " fits files")
        
    print('Start frame =',start_frame)

    if testing:
        if test_number == None:
            raise ValueError("If testing, you must specify the number of frames to test in the test_number keyword.")
        files=files[start_frame:start_frame+int(test_number)]
    else:
        files = files[start_frame:end_frame]
    print('New file count = ', len(files))
    
    print('Reading file headers and creating high pass filtered frames...')

    chops, frame_medians = [], []
    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        chops, frame_medians, para_angles = zip(*tqdm(pool.imap(FileInfoHighPass((highpass_dir, bools, obj, skip_target_check)), files), total=len(files)))
    
    return np.asarray(files), np.asarray(chops), np.asarray(frame_medians), np.asarray(para_angles), highpass_dir

def chop_correction(files, highpass_dir, chops, para_angles, nbg=5, framew=512, frameh=512, coadd_limit = 10,\
                    chop_direction = 'UP-DOWN', set_chop_via_bg=False):
    #NBG see above must be odd, greater than or equal to 5
    
    print("Finding chop positions....")
    
    root_dir = os.path.dirname(highpass_dir)

    coadd_dir=os.path.join(root_dir,'coadd')
    
    if not os.path.exists(coadd_dir):
        os.makedirs(coadd_dir)
    
    frames = np.ones((nbg, framew, frameh))

    for j in range(nbg):
        hdul = fits.open(os.path.join(highpass_dir, "highpass_"+files[j].name))
        frames[j] = hdul[0].data
        hdul.close()
        
    lastChopPosition = ""
    chopFreeze = False
    
    for i in tqdm(range(len(files))):
                
        if (i > int(np.floor(nbg/2))) and (i < (len(files) - int(np.floor(nbg/2)))):
            hdul = fits.open(os.path.join(highpass_dir, "highpass_"+files[nbg+i - int(np.floor(nbg/2)) -1].name))
            frames = np.concatenate((frames[1:], [hdul[0].data]))
            hdul.close()
    
        if chopFreeze == False:
            bg =  np.min(frames, axis=0)
    
        image = fits.open(os.path.join(highpass_dir, "highpass_"+files[i].name))
        
        subtracted = image[0].data - bg
    
        indices = np.where(subtracted == np.nanmax(subtracted))
        
        if not set_chop_via_bg:
    
            if (chop_direction == "UP-DOWN") or (chop_direction == "DIAGONAL"):
                if indices[0][0] < framew/2:
                    chops[i] = "CHOP_A"
                else:
                    chops[i] = "CHOP_B"
            elif (chop_direction == "LEFT-RIGHT"):
                if indices[1][0] < frameh/2:
                    chops[i] = "CHOP_A"
                else:
                    chops[i] = "CHOP_B"     
        else:
            raise ValueError("Code for setting chop via background is not written yet")
    
        if chops[i] == lastChopPosition:
            chopFreeze = True
        else:
            chopFreeze = False
        lastChopPosition = chops[i]
        
        image.close()

    orig_chops = np.copy(chops)

    print("Finding consecutive repeat chop positions...")
    
    chop_groups = []
    k = 0
    for i, j in tqdm(groupby(chops)):
        sum = (len(list(j)))
        if sum != 1:
            chop_groups.append((k, sum))
        k += sum
    
    print("Coadding consecutive repeat chop positions...")

    for group in tqdm(chop_groups):
        frames = np.zeros((1, framew, frameh))
        count = 0
        for i in range(group[1]):
            hdul = fits.open(files[group[0]+i])
            if (i < coadd_limit):
                count += 1
                frames += hdul[0].data
            hdul.close()
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

    return files, chops, para_angles, orig_chops

def subtract_background(files, raw_dir, flat=None, framew=512, frameh=512, edge_cut = 2, channel_edges=[127, 255, 383], vertical_lines=[303],\
                        threadcount=50):
    
    print("Subtracting backgrounds....")

    root_dir = os.path.dirname(raw_dir)

    subtracted_dir=os.path.join(root_dir,'subtracted')

    if not os.path.exists(subtracted_dir):
        os.makedirs(subtracted_dir)
    
    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        minima, maxima = zip(*tqdm(pool.imap(SubtractBackground((subtracted_dir, files, framew, frameh, edge_cut,\
                                                    channel_edges, vertical_lines, flat)), range(len(files))), total=len(files)))

    return subtracted_dir 

def frame_registration(files, subtracted_dir, windowsize=20, nan_mask_size=0.4, threadcount=50):
    
    print("Aligning frames....")    
    
    root_dir = os.path.dirname(subtracted_dir)
    aligned_dir = os.path.join(root_dir, 'aligned')
    if not os.path.exists(aligned_dir):
        os.makedirs(aligned_dir)
        
    wx = np.linspace(0, 2*windowsize-1, 2*windowsize)
    wy = np.linspace(0, 2*windowsize-1, 2*windowsize)
    wx, wy = np.meshgrid(wx, wy)

    hdul = fits.open(os.path.join(subtracted_dir, "subtracted_"+files[0].name))
    hdul2 = fits.open(os.path.join(subtracted_dir, "subtracted_"+files[1].name))
    
    first_frame = hdul[0].data
    second_frame = hdul2[0].data

    first_maxima = np.where(first_frame == np.nanmax(first_frame))
    second_maxima = np.where(second_frame == np.nanmax(second_frame))
    first_minima = np.where(first_frame == np.nanmin(first_frame))
    second_minima = np.where(second_frame == np.nanmin(second_frame))

    hdul.close()
    hdul2.close()

    frameh, framew = np.shape(first_frame)
    
    x = np.linspace(0, framew-1, framew)
    y = np.linspace(0, frameh-1, frameh)
    
    first_cutout = first_frame[(first_maxima[0][0]-windowsize):(first_maxima[0][0]+windowsize),(first_maxima[1][0]-windowsize):(first_maxima[1][0]+windowsize)]
    second_cutout = second_frame[(second_maxima[0][0]-windowsize):(second_maxima[0][0]+windowsize),(second_maxima[1][0]-windowsize):(second_maxima[1][0]+windowsize)]
    
    reffit, _ = curve_fit(Circular_Gaussian2D_ravel, (wx, wy), first_cutout.ravel(), p0=[np.max(first_cutout),\
                        10, -1, windowsize, windowsize],\
                       bounds=([0, 1, 1*-np.inf, 1, 1], [2*np.max(first_cutout), 200, np.inf, 2*windowsize, 2*windowsize]))
    
    reference = Circular_Gaussian2D((wx, wy), reffit[0], reffit[1], reffit[2], windowsize-0.5, windowsize-0.5)

    first_offset_x, first_offset_y = chi2_shift(reference, first_cutout, upsample_factor='auto', return_error=False)  
    second_offset_x, second_offset_y = chi2_shift(reference, second_cutout, upsample_factor='auto', return_error=False)  
    
    second_offset_x += (second_maxima[1][0] - first_maxima[1][0])
    second_offset_y += (second_maxima[0][0] - first_maxima[0][0])
    
    padding_x = int(np.ceil(np.abs(second_offset_x))*np.sign(second_offset_x))
    padding_y = int(np.ceil(np.abs(second_offset_y))*np.sign(second_offset_y))
    
    px = np.linspace(0, framew-1+np.abs(padding_x), framew+np.abs(padding_x))
    py = np.linspace(0, frameh-1+np.abs(padding_y), frameh+np.abs(padding_y))

    first_frame[circular_mask((first_minima[1][0], first_minima[0][0]), nan_mask_size*reffit[1], framew, frameh)] = np.nan
    second_frame[circular_mask((second_minima[1][0], second_minima[0][0]), nan_mask_size*reffit[1], framew, frameh)] = np.nan
        
    first_frame = align_frame(first_frame, px, py, padding_x, padding_y, first_offset_x, first_offset_y)
    
    origin = [first_maxima[1][0]-0.5, first_maxima[0][0]-0.5]
    
    if padding_y >= 0:
        origin[1] = origin[1] + padding_y
    if padding_x >= 0:
        origin[0] = origin[0] + padding_x

    center_padding_x = -1*int(2*origin[0] - np.shape(first_frame)[1] + 1)
    center_padding_y = -1*int(2*origin[1] - np.shape(first_frame)[0] + 1)

    first_frame = pad_frame(first_frame, len(px) + np.abs(center_padding_x), len(py) + np.abs(center_padding_y), center_padding_x, center_padding_y)

    newhdul = fits.HDUList([fits.PrimaryHDU(data=(first_frame))])
    newhdul.writeto(os.path.join(aligned_dir, "aligned_"+files[0].name), overwrite=True)
    
    second_frame = align_frame(second_frame, px, py, padding_x, padding_y, second_offset_x, second_offset_y)
    second_frame = pad_frame(second_frame, len(px) + np.abs(center_padding_x), len(py) + np.abs(center_padding_y), center_padding_x, center_padding_y)
   
    newhdul = fits.HDUList([fits.PrimaryHDU(data=(second_frame))])   
    newhdul.writeto(os.path.join(aligned_dir, "aligned_"+files[1].name), overwrite=True)
    newhdul.close()

    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        psfmaxima, background_dev = zip(*tqdm(pool.imap(RegisterFrames(( subtracted_dir,\
                    aligned_dir, padding_x, padding_y, center_padding_x, center_padding_y, px, py,\
                                reference, first_maxima, windowsize, nan_mask_size*reffit[1])), files[2:]), total=len(files) - 2))
            
    psfmaxima, background_dev = np.concatenate((np.zeros(2), np.asarray(psfmaxima))), np.concatenate((np.zeros(2), np.asarray(background_dev)))

    psfmaxima[0] = np.nanmax(first_frame)
    psfmaxima[1] = np.nanmax(second_frame)
    background_dev[0] = np.nanstd(first_frame[np.abs(first_frame - np.nanmedian(first_frame)) < 3*np.nanstd(first_frame)])
    background_dev[1] = np.nanstd(second_frame[np.abs(second_frame - np.nanmedian(second_frame)) < 3*np.nanstd(second_frame)])
    
    aligned_files = np.asarray(sorted(list(pathlib.Path(str(aligned_dir)).rglob('*.fits'))))
    
    return psfmaxima, background_dev, reffit, np.shape(first_frame), float(first_frame.nbytes), aligned_files

def frame_evaluation(aligned_files, chops, array_shape, file_size, tolerance=0.9, pxscale=0.0179, windowsize=20, threadcount=50, memoryMode=1):
    
    wx = np.linspace(0, 2*windowsize-1, 2*windowsize)
    wy = np.linspace(0, 2*windowsize-1, 2*windowsize)
    wx, wy = np.meshgrid(wx, wy)
    
    if memoryMode == 0:
        
        chopa_mean_frame = np.nanmean(aligned_files[chops == "CHOP_A"], axis=0)
        chopb_mean_frame = np.nanmean(aligned_files[chops == "CHOP_B"], axis=0)

        chopb_mean_frame[np.isnan(chopb_mean_frame)] = 0
        chopa_mean_frame[np.isnan(chopa_mean_frame)] = 0

        
        #if __name__ == "__main__":
        with Pool(threadcount) as pool:
            psfmaxima, background_dev, correlations, amplitudes, sigmax, sigmay, gauss_offsets =\
                zip(*tqdm(pool.imap(EvaluateFramesMem((aligned_files, chopa_mean_frame, chopb_mean_frame, wx, wy, windowsize, array_shape)),\
                                    np.array((chops, np.arange(len(aligned_files)))).T), total=len(aligned_files)))
        
    else:
    
        stats = psutil.virtual_memory()  # returns a named tuple
        available = float(getattr(stats, 'available'))
        
        chopa_files = aligned_files[chops == "CHOP_A"]
        chopb_files = aligned_files[chops == "CHOP_B"]
        
        a_buffer = int(np.ceil((file_size*threadcount*len(chopa_files))/(tolerance*available)))
        b_buffer = int(np.ceil((file_size*threadcount*len(chopb_files))/(tolerance*available)))
    
        print("Creating integrated files for correlation...")
        print("Using a buffer of ", int(len(chopa_files)/a_buffer), " frames...")
    
        a_splitlist = np.linspace(0, len(chopa_files), 1+a_buffer)[1:-1].round().astype(int)
        a_filebufs = np.split(chopa_files, a_splitlist)
        
        b_splitlist = np.linspace(0, len(chopb_files), 1+b_buffer)[1:-1].round().astype(int)
        b_filebufs = np.split(chopb_files, b_splitlist)
    
    
        #if __name__ == "__main__":
        with Pool(threadcount) as pool:
            a_bigarr, a_filecounts = zip(*tqdm(pool.imap(IntegrateFrames((array_shape)), a_filebufs),\
                                               desc="integrating files", total=len(a_filebufs)))
        
        #if __name__ == "__main__":
        with Pool(threadcount) as pool:
            b_bigarr, b_filecounts = zip(*tqdm(pool.imap(IntegrateFrames((array_shape)), b_filebufs),\
                                               desc="integrating files", total=len(b_filebufs)))
        
        chopa_mean_frame = np.sum(a_bigarr, axis=0)/np.sum(a_filecounts)
        
        chopb_mean_frame = np.sum(b_bigarr, axis=0)/np.sum(b_filecounts)

        chopb_mean_frame[np.isnan(chopb_mean_frame)] = 0
        chopa_mean_frame[np.isnan(chopa_mean_frame)] = 0

        #if __name__ == "__main__":
        with Pool(threadcount) as pool:
            psfmaxima, background_dev, correlations, amplitudes, sigmax, sigmay, gauss_offsets =\
                zip(*tqdm(pool.imap(EvaluateFrames((chopa_mean_frame, chopb_mean_frame, wx, wy, windowsize, array_shape)),\
                                    np.array((chops, aligned_files)).T), total=len(aligned_files)))
    
    sigmax, sigmay = np.asarray(sigmax), np.asarray(sigmay)

    fwhms = 2*np.sqrt(2*np.log(2)*(sigmax+sigmay))*pxscale
    eccentricities = np.sqrt(1 - sigmax/sigmay)
    eccentricities_r = np.sqrt(1 - sigmay/sigmax)
    eccentricities[np.isnan(eccentricities)] = eccentricities_r[np.isnan(eccentricities)]
    
    return fwhms, eccentricities, np.asarray(psfmaxima), np.asarray(background_dev), np.asarray(correlations), np.asarray(amplitudes), np.asarray(gauss_offsets), sigmax, sigmay

def frame_rejection(chops, params, sigma=None):

    chopa_bool = (chops == "CHOP_A")
    chopb_bool = (chops == "CHOP_B")

    if sigma is None:
        sigma = 1.5*np.ones(len(params))

    for i in range(len(params)):
        
        params[i][chopa_bool] *= np.nanstd(params[i][chopb_bool])/np.nanstd(params[i][chopa_bool])
        params[i][chopa_bool] += np.nanmedian(params[i][chopb_bool]) - np.nanmedian(params[i][chopa_bool])

    bools = ~np.isnan(params[0])

    for i in range(len(params)):
        bools = bools & (params[i] < np.nanmedian(params[i]) + sigma[i]*np.nanstd(params[i]))

    return bools
    
'''
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
'''

def frame_binning(aligned_files, raw_dir, frame_bool, chops, para_angles, array_shape, bin=50, threadcount=50, memoryMode=1):
    
    root_dir = os.path.dirname(raw_dir)
    binned_dir=os.path.join(root_dir,'binned')

    if not os.path.exists(binned_dir):
        os.makedirs(binned_dir)

    chopa_files = aligned_files[frame_bool & (chops == "CHOP_A")]
    chopb_files = aligned_files[frame_bool & (chops == "CHOP_B")]
    chopa_angles = para_angles[frame_bool & (chops == "CHOP_A")]
    chopb_angles = para_angles[frame_bool & (chops == "CHOP_B")]
    
    a_buffer = int(np.ceil(len(chopa_files)/bin))
    b_buffer = int(np.ceil(len(chopb_files)/bin))
    
    a_splitlist = np.linspace(0, len(chopa_files), 1+a_buffer)[1:-1].round().astype(int)
    a_binfiles = np.split(chopa_files, a_splitlist)
    a_angles = np.split(chopa_angles, a_splitlist)
    b_splitlist = np.linspace(0, len(chopb_files), 1+b_buffer)[1:-1].round().astype(int)
    b_binfiles = np.split(chopb_files, b_splitlist)
    b_angles = np.split(chopb_angles, b_splitlist)

    print("Binning files...")
    
    numbinfiles = len(a_binfiles) + len(b_binfiles)

    binned_chops = np.empty(numbinfiles, dtype="<U16")
    binned_angles = np.zeros(numbinfiles)

    if memoryMode == 0:

        binned_frames = np.zeros((numbinfiles, array_shape[0], array_shape[1]))
        
        #if __name__ == "__main__":
        with Pool(threadcount) as pool:
            a_binned_frames, a_binned_angles, a_binned_filenames  = zip(*tqdm(pool.imap(BinFramesMem((array_shape)), zip(a_angles, a_binfiles))))

        #if __name__ == "__main__":
        with Pool(threadcount) as pool:
            b_binned_frames, b_binned_angles, b_binned_filenames = zip(*tqdm(pool.imap(BinFramesMem((array_shape)), zip(b_angles, b_binfiles))))
            
        binned_filenames = np.asarray(sorted(a_binned_filenames+b_binned_filenames))
        a_binned_frames, a_binned_angles, a_binned_filenames = np.asarray(a_binned_frames), np.asarray(a_binned_angles), np.asarray(a_binned_filenames)
        b_binned_frames, b_binned_angles, b_binned_filenames = np.asarray(b_binned_frames), np.asarray(b_binned_angles), np.asarray(b_binned_filenames)

        binned_chops[np.where(np.isin(binned_filenames,a_binned_filenames) == True)[0]] = "CHOP_A"
        binned_chops[np.where(np.isin(binned_filenames,b_binned_filenames) == True)[0]] = "CHOP_B"
    
        for i in range(numbinfiles):
            if binned_filenames[i] in a_binned_filenames:
                binned_angles[i] = a_binned_angles[np.where(a_binned_filenames == binned_filenames[i])[0]]
                binned_frames[i] = a_binned_frames[np.where(a_binned_filenames == binned_filenames[i])[0]]
            else:
                binned_angles[i] = b_binned_angles[np.where(b_binned_filenames == binned_filenames[i])[0]]
                binned_frames[i] = b_binned_frames[np.where(b_binned_filenames == binned_filenames[i])[0]]
        
        return binned_frames, binned_chops, binned_angles
        
    else:
            
        #if __name__ == "__main__":
        with Pool(threadcount) as pool:
            a_binned_angles, a_binned_filenames  = zip(*tqdm(pool.imap(BinFrames((array_shape, binned_dir)), zip(a_angles, a_binfiles))))
        a_binned_angles, a_binned_filenames = np.asarray(a_binned_angles), np.asarray(a_binned_filenames)
        
        #if __name__ == "__main__":
        with Pool(threadcount) as pool:
            b_binned_angles, b_binned_filenames = zip(*tqdm(pool.imap(BinFrames((array_shape, binned_dir)), zip(b_angles, b_binfiles))))
        b_binned_angles, b_binned_filenames = np.asarray(b_binned_angles), np.asarray(b_binned_filenames)

        binned_files = np.asarray(sorted(list(pathlib.Path(str(binned_dir)).rglob('*.fits'))))
        binned_filenames = np.asarray([f.name for f in binned_files])
      
        binned_chops[np.where(np.isin(binned_filenames,a_binned_filenames) == True)[0]] = "CHOP_A"
        binned_chops[np.where(np.isin(binned_filenames,b_binned_filenames) == True)[0]] = "CHOP_B"
    
        for i in range(len(binned_files)):
            if binned_filenames[i] in a_binned_filenames:
                binned_angles[i] = a_binned_angles[np.where(a_binned_filenames == binned_filenames[i])[0]]
            else:
                binned_angles[i] = b_binned_angles[np.where(b_binned_filenames == binned_filenames[i])[0]]
        
        return binned_files, binned_chops, binned_angles

    
def create_stacked_flat(files, chops, tolerance=0.9, framew=512, frameh=512, threadcount=50):
    
    stats = psutil.virtual_memory()  # returns a named tuple
    available = float(getattr(stats, 'available'))

    hdul = fits.open(files[0])
    file_size = float(hdul[0].data[0].nbytes)
    hdul.close()
    
    chopa_files = files[chops == "CHOP_A"]
    chopb_files = files[chops == "CHOP_B"]
    
    a_buffer = int(np.ceil((file_size*threadcount*len(chopa_files))/(tolerance*available)))
    b_buffer = int(np.ceil((file_size*threadcount*len(chopb_files))/(tolerance*available)))

    print("Creating integrated files for correlation...")
    print("Using a buffer of ", int(len(chopa_files)/a_buffer), " frames...")

    a_splitlist = np.linspace(0, len(chopa_files), 1+a_buffer)[1:-1].round().astype(int)
    a_filebufs = np.split(chopa_files, a_splitlist)
    
    b_splitlist = np.linspace(0, len(chopb_files), 1+b_buffer)[1:-1].round().astype(int)
    b_filebufs = np.split(chopb_files, b_splitlist)

    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        a_bigarr, a_filecounts = zip(*tqdm(pool.imap(IntegrateFrames(((framew, frameh))), a_filebufs),\
                                           desc="integrating files", total=len(a_filebufs)))
    
    chopa_mean_frame = np.sum(a_bigarr, axis=0)/np.sum(a_filecounts)
    
    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        b_bigarr, b_filecounts = zip(*tqdm(pool.imap(IntegrateFrames(((framew, frameh))), b_filebufs),\
                                           desc="integrating files", total=len(b_filebufs)))
    
    chopb_mean_frame = np.sum(b_bigarr, axis=0)/np.sum(b_filecounts)

    flat = np.concatenate((chopb_mean_frame[:int(framew/2), :], chopa_mean_frame[int(framew/2):, :]))
    flat[flat == 0] = np.min(flat[flat != 0]) 
    return flat
    
def inject_planet(binned_files, binned_amps, binned_sigmax, binned_sigmay, binned_angles, array_shape, crop_size = None, highpassrad=None, radius=80, theta=45, ratio=0.01, file_save=False, threadcount=50):

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
        
    wx, wy = np.meshgrid(y, x)

    if binned_files.ndim == 3:
        stacked = np.vstack((np.arange(len(binned_files)), binned_amps, binned_sigmax, binned_sigmay, binned_angles)).T
    else:
        stacked = np.vstack((binned_files, binned_amps, binned_sigmax, binned_sigmay, binned_angles)).T
        
    if threadcount != 0:
        
        if not file_save:
    
            #if __name__ == "__main__":
            with Pool(threadcount) as pool:
                injected_array, check_array = zip(*pool.imap(InjectPlanetMem((binned_files, wx, wy, array_shape, radius, theta, ratio, width, height, highpassrad)), stacked))
    
            sorted_injections = np.asarray(injected_array)[np.argsort(check_array)]
                
            return sorted_injections, array_shape
            
        else:
    
            root_dir = os.path.dirname(os.path.dirname(binned_files[0]))
            injected_dir = os.path.join(root_dir, 'injected')
            if not os.path.exists(injected_dir):
                os.makedirs(injected_dir)
    
            #if __name__ == "__main__":
            with Pool(threadcount) as pool:
                check_array = zip(*tqdm(pool.imap(InjectPlanet((wx, wy, injected_dir, array_shape, radius, theta, ratio, width, height, highpassrad)), stacked), total=len(stacked)))
                
            injected_files = np.asarray(sorted(list(pathlib.Path(str(injected_dir)).rglob('*.fits'))))
        
            return injected_files, array_shape
    
    else:

        injected_array = np.zeros((len(stacked), array_shape[0], array_shape[1]))

        for i in range(len(stacked)):

            if binned_files.ndim == 3:

                img = binned_files[i][width[0]:width[1], height[0]:height[1]]

            else:
                
                info = stacked[i]
                hdul = fits.open(info[0])
                img = hdul[0].data[width[0]:width[1], height[0]:height[1]]
                hdul.close()
        
            planet = Gaussian2D((wx, wy), ratio*info[1],info[2],info[3], 0,\
                 0.5*(array_shape[0] - 1) + radius*np.cos((theta - info[4])*np.pi/180), 0.5*(array_shape[1] - 1) + radius*np.sin((theta - info[4])*np.pi/180))

            img += planet

            if highpassrad != None:

                max_indices = np.where(img == np.nanmax(img))
                maximum = (max_indices[1][0], max_indices[0][0])
                
                max_mask =  circular_mask((maximum[0], maximum[1]), 2.2*hwhm, array_shape[1], array_shape[0])
                max_aperture =  circular_mask((maximum[0], maximum[1]), 2.2*1.1*hwhm, array_shape[1], array_shape[0]) ^ max_mask
                
                new_bg = np.copy(img)
                new_bg[max_mask] = np.median(new_bg[max_aperture])
                
                img = img - convolve_fft(np.pad(new_bg, 50, mode='edge'), Ring2DKernel(int(highpassrad*5/4), highpassrad))[50:-50, 50:-50]

            injected_array[i] = img

        return injected_array, array_shape


def forward_model(binned_files, binned_amps, binned_sigmax, binned_sigmay, binned_angles, array_shape, aperture_size, \
                    injection_radius=45, injection_angle=0, initial_contrast=8e-4, delta_rot=0, n_segments=1, ncomp=5, crop_size=None, highpassrad=None, file_save=False, threadcount=30, nproc=None):
        
        data_input, injected_shape = inject_planet(binned_files, binned_amps, binned_sigmax,\
                                                       binned_sigmay, binned_angles, array_shape,\
                                                       threadcount=30, ratio=initial_contrast, radius=injection_radius,\
                                                       theta=injection_angle, crop_size=crop_size, file_save=file_save, highpassrad=highpassrad)
    

        origin = [injected_shape[0]/2 - 0.5, injected_shape[1]/2 - 0.5]
        IWA_mask =  circular_mask(origin, aperture_size, injected_shape[0], injected_shape[1])
        '''
        med_image_unsubtracted = np.nanmedian(data_input, axis=0)
        stellar_flux = np.nansum(med_image_unsubtracted[IWA_mask])
        stellar_peak_flux = np.nanmax(med_image_unsubtracted[IWA_mask])
        '''
        if delta_rot == 0 and n_segments <= 1:
        
            med_image = pca_annulus(data_input, binned_angles, ncomp=ncomp, annulus_width=1.05*2*aperture_size, r_guess=injection_radius, delta_rot=delta_rot, n_segments=n_segments, nproc=nproc, svd_mode='eigen', imlib='opencv')
        
        else:
            
            med_image = pca_annular(data_input, binned_angles, fwhm=aperture_size*2, ncomp=ncomp, asize=1.05*2*aperture_size, radius_int=injection_radius-aperture_size, verbose=False,
                               nproc=nproc, svd_mode='eigen', imlib='opencv')
    
        circle_pos = np.linspace(0, 2*np.pi, int(np.floor(np.pi*injection_radius/aperture_size))+1)[:-1]
        fluxes = np.zeros(len(circle_pos))
        
        for i in range(len(circle_pos)):
        
            mask = circular_mask((origin[0] + injection_radius*np.cos(circle_pos[i] + injection_angle*np.pi/180), origin[1] +\
                                  injection_radius*np.sin(circle_pos[i] + injection_angle*np.pi/180)), aperture_size,\
                                 np.shape(med_image)[0], np.shape(med_image)[1])
            '''
            display = np.copy(med_image)
            display[mask]  = 0
            plt.imshow(display, cmap="inferno")
            plt.show()
            '''
            if i == 0:

                planet_peak_flux = (np.nanmax(med_image[mask]))
                
            fluxes[i] = np.nansum(med_image[mask])

        noise_factor = (np.std(fluxes[1:])*np.sqrt(1 + 1/(len(fluxes) - 1)))

        snr = (fluxes[0] - np.mean(fluxes[1:]))/noise_factor
        
        #measured_contrast = initial_contrast + noise_factor*(5 - snr)/stellar_flux#((stellar_peak_flux/planet_peak_flux)*fluxes[0])

        return snr, noise_factor


def contrast_curve_parallelized(binned_files, binned_amps, binned_sigmax, binned_sigmay, binned_angles, curve_prior, array_shape,\
                        injection_radii, angles, hwhm, file_save=False, tolerance=0.05, high_buffer = 2, delta_rot = 0, n_segments=1, max_iterations = 10, highpassrad=None, ncomp=5, add_info=[None, None], threadcount=6):

    coordlist = np.vstack(np.asarray(np.meshgrid(angles, injection_radii)).T)
    
    with Pool(threadcount) as pool:
        approx_contrast_list, snrs_list = zip(*tqdm(pool.imap\
            (GetContrasts((binned_files, binned_amps, binned_sigmax, binned_sigmay, binned_angles, curve_prior, array_shape,\
                         hwhm, file_save, tolerance, high_buffer, delta_rot, n_segments, max_iterations, highpassrad, ncomp, add_info)), coordlist), total=len(coordlist)))

    #Average over angles
    approx_contrasts = np.asarray(approx_contrast_list).reshape(len(angles), len(injection_radii))
    snrs = np.asarray(snrs_list).reshape(len(angles), len(injection_radii))

    return approx_contrasts, snrs
    
def contrast_curve(binned_files, binned_amps, binned_sigmax, binned_sigmay, binned_angles, curve_prior, array_shape,\
                        injection_radii, angles, hwhm, file_save=False, tolerance=0.05, high_buffer = 2, delta_rot = 0, n_segments=1, max_iterations = 10, highpassrad=None, ncomp=5, add_info=[None, None]):
    
    snrs_list, approx_contrast_list =\
        np.zeros((len(angles)*len(injection_radii))), np.zeros((len(angles)*len(injection_radii)))

    coordlist = np.vstack(np.asarray(np.meshgrid(angles, injection_radii)).T)

    for i in tqdm(range(len(coordlist))):
        
            last_contrast = high_buffer*curve_prior(coordlist[i][1])

            snr_plot = np.array([])
            contrast_plot = np.array([])
                        
            anticorrelation = False
            correlation = False
            lin_error = False
            counter = 0
        
            while True:

                try:
                    
                    curr_snr, curr_noise_factor =\
                        forward_model(binned_files, binned_amps, binned_sigmax, binned_sigmay, binned_angles, array_shape,\
                                      hwhm, injection_radius=coordlist[i][1], injection_angle=coordlist[i][0], initial_contrast=last_contrast, crop_size=2*int(np.ceil(1.05*(coordlist[i][1]+hwhm))), delta_rot=delta_rot, n_segments=n_segments, file_save=file_save, highpassrad=highpassrad, ncomp=ncomp)
                    
                    snr_plot = np.append(snr_plot, curr_snr)
                    contrast_plot = np.append(contrast_plot, last_contrast)
                    #print("Contrast: ",last_contrast)
                    #print("SNR: ",curr_snr)
    
                    if (np.abs(curr_snr - 5)/(5) < tolerance):
                        
                        break
                        
                    else:
                        
                        if np.max(snr_plot) > 5:
                            
                            if (len(snr_plot) > 1):
    
                                results = linregress(snr_plot[-2:], np.log(contrast_plot[-2:]))
                                curr_approx_contrast = np.exp(np.round(results[0]*5 + results[1], 3)) 
                                #print("Linregress: ", curr_approx_contrast)
                                
                                if curr_approx_contrast < 1e-6 or curr_approx_contrast> 1e-1:
                                    
                                    if curr_snr < 0:
                                        
                                        best_snr = snr_plot[1:][np.nanargmin(np.abs(snr_plot[1:] - 5))]
                                        best_contrast = contrast_plot[1:][np.nanargmin(np.abs(snr_plot[1:] - 5))]
                                        snr_plot = snr_plot[:-1]
                                        contrast_plot = contrast_plot[:-1]
                                        
                                    else:
                                        
                                        best_snr = curr_snr
                                        best_contrast = last_contrast
                                    
                                    curr_approx_contrast = np.abs((5/best_snr)*(best_contrast))
                                    if np.isin(curr_approx_contrast, contrast_plot):
                                        
                                        results = linregress(snr_plot[snr_plot > 0], np.log(contrast_plot[snr_plot > 0]))
                                        curr_approx_contrast = np.exp(np.round(results[0]*5 + results[1], 3)) 
                                        #print("LR Linregress: ", curr_approx_contrast)
                                    #print("LR Scaling: ", curr_approx_contrast)
                                '''
                                plt.figure(1)
                                plt.scatter(np.asarray(snr_plot), np.asarray(contrast_plot), s=10, color="orange")
                                plt.plot(snr_plot[-2:], np.log(contrast_plot[-2:]))
                                plt.scatter([5],[curr_approx_contrast],  s=20, color="red")
                                plt.vlines([5], ymin=1e-4, ymax=8e-4, color="green")
                                plt.yscale("log")
                                plt.xlabel("SNR")
                                plt.ylabel("Contrast")
                                plt.show()
                                '''
                                
                            else:
                                
                                curr_approx_contrast = np.abs((5/curr_snr)*(last_contrast))
                        else:
                            if len(snr_plot) > 1:
                                if ((snr_plot[-1] < snr_plot[-2]) & (contrast_plot[-1] > contrast_plot[-2])) or\
                                        ((snr_plot[-1] > snr_plot[-2]) & (contrast_plot[-1] < contrast_plot[-2])):
                                    anticorrelation = True
                                    curr_approx_contrast = 0.25*last_contrast
                                if ((snr_plot[-1] > snr_plot[-2]) & (contrast_plot[-1] > contrast_plot[-2])) or\
                                        ((snr_plot[-1] < snr_plot[-2]) & (contrast_plot[-1] < contrast_plot[-2])):
                                    correlation = True
                                    curr_approx_contrast = 2*last_contrast
                            else:
                                curr_approx_contrast = 2*last_contrast
                                
                        last_contrast = curr_approx_contrast
    
                    counter += 1

                except:
                    lin_error=True
                    np.savez("Error_log-"+"_iteration-"+str(i)+"_binning-"+str(add_info[0])+"_sigma-"+str(add_info[1])+"_ncomp-"+str(ncomp)+"_highpass-"+\
                             str(highpassrad)+".npz", i, add_info[0], add_info[1], ncomp, highpassrad)
                    pass

                if anticorrelation & correlation or counter > max_iterations or lin_error:
                    '''
                    plt.figure(1)
                    plt.scatter(np.asarray(snr_plot), np.asarray(contrast_plot), s=10, color="orange")
                    plt.vlines([5], ymin=1e-4, ymax=8e-4, color="green")
                    plt.yscale("log")
                    plt.xlabel("SNR")
                    plt.ylabel("Contrast")
                    plt.show()
                    '''
                    try:
                        last_contrast = contrast_plot[np.nanargmin(np.abs(snr_plot - 5))]
                        curr_snr = snr_plot[np.nanargmin(np.abs(snr_plot - 5))]
                    except:
                        last_contrast = np.nan
                        curr_snr = np.nan
                        
                    if anticorrelation & correlation:
                        print("Absolute max reached: ", last_contrast)
                    else:
                        print("Max iterations reached: ", last_contrast)
                    break

 
            approx_contrast_list[i] = last_contrast
            snrs_list[i] = curr_snr

    #Average over angles
    approx_contrasts = np.asarray(approx_contrast_list).reshape(len(angles), len(injection_radii))
    snrs = np.asarray(snrs_list).reshape(len(angles), len(injection_radii))
    return approx_contrasts, snrs
        
def repairChannelEdges(image, loc):
    '''
    Fill in data for three horizontal channel edges on the NOMIC detector, to prevent artifacting.
    Each channel edge is three rows tall and requires data for each
    '''
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
    '''
    Repair vertical line artifact in the NOMIC detector while preserving the unique noise profile.
    '''
    gauss = Gaussian1DKernel(stddev=5)
    
    # Average adjacent columns
    true = 0.5*(image[:, loc+1] + image[:, loc-1])

    # Get smoothed difference between the column and the averaged adjacent columns and add it back
    image[:, loc] += convolve(true - image[:, loc], gauss)

    return image

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
    2D Gaussian function, raveled
    '''
    x, y = xy
    return (offset + amp*np.exp(-1*((x-x0)**2/sigmax + (y-y0)**2/sigmay)))

def circular_mask(center, radius, width, height):
    '''
    Creates circular mask of certain radius in an image of certain width and height
    '''
    Y, X = np.ogrid[:height, :width]
    distance = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = distance <= radius
    
    return mask

def pad_frame(frame, new_xlen, new_ylen, padding_x, padding_y):
    '''
    Pad frame while maintaining image center
    '''
    
    if padding_y < 0:
        
        frame = np.concatenate((frame, np.nan*np.ones((-1*padding_y, new_xlen - np.abs(padding_x)))))
        
    else:
        
        frame = np.concatenate((np.nan*np.ones((padding_y, new_xlen - np.abs(padding_x))), frame))

    if padding_x < 0:
        
        frame = np.concatenate((frame, np.nan*np.ones((new_ylen, -1*padding_x))), axis=1)

    else:

        frame = np.concatenate((np.nan*np.ones((new_ylen, padding_x)), frame), axis=1)

    return frame
    

def align_frame(frame, px, py, padding_x, padding_y, offset_x, offset_y):
    '''
    Aligns frame
    '''
    frame = pad_frame(frame, len(px), len(py), padding_x, padding_y)
    interp = RegularGridInterpolator((py, px), frame, bounds_error=False)
    Y, X = np.meshgrid(px+offset_x, py+offset_y)
    
    newimage = interp((X, Y))
    return newimage
