#----------------------------------------
# IMPORTS
#----------------------------------------

import numpy as np
import os, pathlib
from tqdm.auto import tqdm
from astropy.convolution import convolve_fft, Ring2DKernel
from scipy.stats import linregress
from multiprocessing.pool import ThreadPool as Pool
from vip_pca_functions import pca_annular, pca_annulus

#----------------------------------------
# CLASSES
#----------------------------------------

class InjectPlanet(object):

    def __init__(self, params):
        self.params = params
    
    def __call__(self, info):

            binned_frames, directory, wx, wy, array_shape, radius, theta, ratio, width, height, highpassrad = self.params

            planet = Gaussian2D((wx, wy), ratio*info[1],info[2],info[3], 0,\
                 0.5*(array_shape[0] - 1) + radius*np.cos((theta - info[4])*np.pi/180), 0.5*(array_shape[1] - 1) + radius*np.sin((theta - info[4])*np.pi/180))

            if binned_frames.ndim == 3:

                img = binned_frames[int(info[0])][width[0]:width[1], height[0]:height[1]] + planet

            else:
                
                hdul = fits.open(info[0])
                img = hdul[0].data[width[0]:width[1], height[0]:height[1]] + planet
                hdul.close()

            if highpassrad is not None:
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


            if directory is not None:
            
                newhdul = fits.HDUList([fits.PrimaryHDU(data=(img))])      
                newhdul.writeto(os.path.join(directory, "injected_"+info[0].name), overwrite=True)
                newhdul.close()

                return True, True

            else:
    
                return img, info[0]


class GetContrasts(object):

        def __init__(self, params):
            self.params = params
        
        def __call__(self, coords):

                binned_frames, binned_amps, binned_sigmax, binned_sigmay, binned_angles, curve_prior, array_shape,\
                         hwhm, file_save, tolerance, high_buffer, delta_rot, n_segments, max_iterations, highpassrad, ncomp, add_info = self.params                    
    
                last_contrast = high_buffer*curve_prior(coords[1])
    
                snr_plot = np.array([])
                contrast_plot = np.array([])
                            
                anticorrelation = False
                correlation = False
                lin_error = False
                counter = 0
            
                while True:
                 
                    try:
                        
                            curr_snr, curr_noise_factor =\
                                forward_model(binned_frames, binned_amps, binned_sigmax, binned_sigmay, binned_angles, array_shape,\
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
                        np.savez("Error_log-"+"_radius-"+str(coords[1])+"_angle-"+str(coords[0])+"_binning-"+str(add_info[0])+"_sigma-"+str(add_info[1])+"_ncomp-"+str(ncomp)+"_highpass-"+\
                                 str(highpassrad)+".npz", coords, add_info[0], add_info[1], ncomp, highpassrad)
                        pass
    
            
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

#----------------------------------------
# FUNCTIONS
#----------------------------------------

def inject_planet(binned_frames, binned_amps, binned_sigmax, binned_sigmay, binned_angles, array_shape, crop_size = None, highpassrad=None, radius=80, theta=45, ratio=0.01, file_save=False, threadcount=50):

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

    if binned_frames.ndim == 3:
        stacked = np.vstack((np.arange(len(binned_frames)), binned_amps, binned_sigmax, binned_sigmay, binned_angles)).T
        first_arg = binned_frames
    else:
        stacked = np.vstack((binned_frames, binned_amps, binned_sigmax, binned_sigmay, binned_angles)).T
        first_arg = None
        
    if file_save:
            root_dir = os.path.dirname(os.path.dirname(binned_frames[0]))
            injected_dir = os.path.join(root_dir, 'injected')
            if not os.path.exists(injected_dir):
                os.makedirs(injected_dir)
    else:
        injected_dir = None
    
    if threadcount != 0:
        
        #if __name__ == "__main__":
        with Pool(threadcount) as pool:
                injected_array, check_array = zip(*pool.imap(InjectPlanet((first_arg, injected_dir, wx, wy, array_shape, radius, theta, ratio, width, height, highpassrad)), stacked))
        
        if file_save:
            
            injected_files = np.asarray(sorted(list(pathlib.Path(str(injected_dir)).rglob('*.fits'))))
        
            return injected_files, array_shape
            
        else:
                
            sorted_injections = np.asarray(injected_array)[np.argsort(check_array)]
                
            return sorted_injections, array_shape
    
    else:

        planet_gen = InjectPlanet((first_arg, injected_dir, wx, wy, array_shape, radius, theta, ratio, width, height, highpassrad))

        
        injected_array = np.zeros((len(stacked), array_shape[0], array_shape[1]))

        if file_save:        
        
            for i in range(len(stacked)):
                
                check, _ = planet_gen(stacked[i])
                
            injected_files = np.asarray(sorted(list(pathlib.Path(str(injected_dir)).rglob('*.fits'))))
        
            return injected_files, array_shape
            
        else:
            
            for i in range(len(stacked)):
                
                injected_array[i], check = planet_gen(stacked[i])

            return injected_array, array_shape


def forward_model(binned_frames, binned_amps, binned_sigmax, binned_sigmay, binned_angles, array_shape, aperture_size, \
                    injection_radius=45, injection_angle=0, initial_contrast=8e-4, delta_rot=0, n_segments=1, ncomp=5, crop_size=None, highpassrad=None, file_save=False, threadcount=30, nproc=None):
        
    data_input, injected_shape = inject_planet(binned_frames, binned_amps, binned_sigmax,\
                                                   binned_sigmay, binned_angles, array_shape,\
                                                   threadcount=30, ratio=initial_contrast, radius=injection_radius,\
                                                   theta=injection_angle, crop_size=crop_size, file_save=file_save, highpassrad=highpassrad)


    origin = [injected_shape[0]/2 - 0.5, injected_shape[1]/2 - 0.5]
    IWA_mask =  circular_mask(origin, aperture_size, injected_shape[1], injected_shape[0])
    '''
    med_image_unsubtracted = np.nanmedian(data_input, axis=0)
    stellar_flux = np.nansum(med_image_unsubtracted[IWA_mask])
    stellar_peak_flux = np.nanmax(med_image_unsubtracted[IWA_mask])
    '''
    if delta_rot == 0 and n_segments <= 1:
    
        med_image = pca_annulus(data_input, binned_angles, ncomp=ncomp, annulus_width=1.05*2*aperture_size, r_guess=injection_radius, delta_rot=delta_rot, n_segments=n_segments,\
                                nproc=nproc, svd_mode='eigen', imlib='opencv')
    
    else:
        
        med_image = pca_annular(data_input, binned_angles, fwhm=aperture_size*2, ncomp=ncomp, asize=1.05*2*aperture_size, radius_int=injection_radius-aperture_size, verbose=False,
                            delta_rot=delta_rot, n_segments=n_segments, nproc=nproc, svd_mode='eigen', imlib='opencv')

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


def contrast_curve_parallelized(binned_frames, binned_amps, binned_sigmax, binned_sigmay, binned_angles, curve_prior, array_shape,\
                        injection_radii, angles, hwhm, file_save=False, tolerance=0.05, high_buffer = 2, delta_rot = 0, n_segments=1, max_iterations = 10, highpassrad=None, ncomp=5, add_info=[None, None], threadcount=6):

    coordlist = np.vstack(np.asarray(np.meshgrid(angles, injection_radii)).T)
    
    with Pool(threadcount) as pool:
        approx_contrast_list, snrs_list = zip(*tqdm(pool.imap\
            (GetContrasts((binned_frames, binned_amps, binned_sigmax, binned_sigmay, binned_angles, curve_prior, array_shape,\
                         hwhm, file_save, tolerance, high_buffer, delta_rot, n_segments, max_iterations, highpassrad, ncomp, add_info)), coordlist), total=len(coordlist)))

    #Average over angles
    approx_contrasts = np.asarray(approx_contrast_list).reshape(len(angles), len(injection_radii))
    snrs = np.asarray(snrs_list).reshape(len(angles), len(injection_radii))

    return approx_contrasts, snrs
    
def contrast_curve(binned_frames, binned_amps, binned_sigmax, binned_sigmay, binned_angles, curve_prior, array_shape,\
                        injection_radii, angles, hwhm, file_save=False, tolerance=0.05, high_buffer = 2, delta_rot = 0, n_segments=1, max_iterations = 10, highpassrad=None, ncomp=5, add_info=[None, None]):
    
    snrs_list, approx_contrast_list =\
        np.zeros((len(angles)*len(injection_radii))), np.zeros((len(angles)*len(injection_radii)))

    coordlist = np.vstack(np.asarray(np.meshgrid(angles, injection_radii)).T)

    contrast_gen = GetContrasts((binned_frames, binned_amps, binned_sigmax, binned_sigmay, binned_angles, curve_prior, array_shape,\
                         hwhm, file_save, tolerance, high_buffer, delta_rot, n_segments, max_iterations, highpassrad, ncomp, add_info))

    for i in tqdm(range(len(coordlist))):
        
        approx_contrast_list[i], snrs_list[i] = contrast_gen(coordlist[i])
        
    #Average over angles
    approx_contrasts = np.asarray(approx_contrast_list).reshape(len(angles), len(injection_radii))
    snrs = np.asarray(snrs_list).reshape(len(angles), len(injection_radii))
    
    return approx_contrasts, snrs
