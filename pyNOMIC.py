#----------------------------------------
# IMPORTS
#----------------------------------------
import numpy as np
from itertools import groupby
import os, pathlib, psutil
import matplotlib.pyplot as plt
from astropy.io import fits
#from astropy.time import Time
from tqdm import tqdm
from astropy.convolution import convolve, convolve_fft,\
    Box2DKernel, Gaussian1DKernel, Ring2DKernel
from scipy.signal import correlate
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
from image_registration import chi2_shift
from multiprocessing.pool import ThreadPool as Pool

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

class FileInfoHighPass(object):

    def __init__(self, params):
        self.params = params
        
    def __call__(self, file):

        highpass_dir, bools, obj, skip_target_check = self.params

        hdul = fits.open(file)

        image = hdul[0].data[0]
        
        # Check if correct object
        if not skip_target_check:
            object = hdul[0].header['OBJNAME']
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
        chop = hdul[0].header['CHOP_POS']

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

        subtracted_dir, files, framew, frameh, edge_cut, channel_edges, vertical_lines = self.params

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
    
        subtracted_frame = unsubtracted[0].data[0] - bg

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
                               bounds=([1, 1, 1, 1*-np.inf, 1, 1], [np.inf, 200, 200, np.inf, 2*windowsize, 2*windowsize]))
        except:
            return corr, np.nan, np.nan, np.nan, np.nan
        return psfmaxima, background_dev, corr, reffit[0], reffit[1], reffit[2], reffit[3]

        
#----------------------------------------
# FUNCTIONS
#----------------------------------------

def combine(obj, raw_dir, master_badmap_dir, combn, side, testing=False, test_number=None, start_frame=0,\
            end_frame = 99e9, skip_target_check=False,background_limit = 28000, threadcount=500):

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
    files = list(pathlib.Path(str(search_dir)).rglob('*.fits'))
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

    return files, chops, para_angles

def subtract_background(files, raw_dir, framew=512, frameh=512, edge_cut = 2, channel_edges=[127, 255, 383], vertical_lines=[303],\
                        threadcount=500):
    
    print("Subtracting backgrounds....")

    root_dir = os.path.dirname(raw_dir)

    subtracted_dir=os.path.join(root_dir,'subtracted')

    if not os.path.exists(subtracted_dir):
        os.makedirs(subtracted_dir)
    
    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        minima, maxima = zip(*tqdm(pool.imap(SubtractBackground((subtracted_dir, files, framew, frameh, edge_cut,\
                                                    channel_edges, vertical_lines)), range(len(files))), total=len(files)))

    return subtracted_dir 

def frame_registration(files, subtracted_dir, windowsize=20, nan_mask_size=0.4, threadcount=500):
    
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
                       bounds=([1, 1, 1*-np.inf, 1, 1], [np.inf, 200, np.inf, 2*windowsize, 2*windowsize]))
    
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
    
    aligned_files = np.asarray(list(pathlib.Path(str(aligned_dir)).rglob('*.fits')))
    
    return psfmaxima, background_dev, reffit, np.shape(first_frame), float(first_frame.nbytes), aligned_files

def frame_evaluation(aligned_files, chops, array_shape, file_size, tolerance=0.7, pxscale=0.0179, windowsize=20, threadcount=500):
    
    stats = psutil.virtual_memory()  # returns a named tuple
    available = float(getattr(stats, 'available'))
    
    chopa_files = aligned_files[chops == "CHOP_A"]
    chopb_files = aligned_files[chops == "CHOP_B"]
    
    a_buffer = int(np.ceil((file_size*len(chopa_files))/(0.7*available)))
    b_buffer = int(np.ceil((file_size*len(chopb_files))/(0.7*available)))
    
    a_splitlist = np.linspace(0, len(chopa_files), 1+a_buffer)[1:-1].round().astype(int)
    a_filebufs = np.split(chopa_files, a_splitlist)
    
    b_splitlist = np.linspace(0, len(chopb_files), 1+b_buffer)[1:-1].round().astype(int)
    b_filebufs = np.split(chopb_files, b_splitlist)

    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        a_bigarr, a_filecounts = zip(*tqdm(pool.imap(IntegrateFrames((array_shape)), a_filebufs), desc="integrating files"))
    
    chopa_mean_frame = np.sum(a_bigarr, axis=0)/np.sum(a_filecounts)
    
    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        b_bigarr, b_filecounts = zip(*tqdm(pool.imap(IntegrateFrames((array_shape)), b_filebufs), desc="integrating files"))
    
    chopb_mean_frame = np.sum(b_bigarr, axis=0)/np.sum(b_filecounts)

    chopb_mean_frame[np.isnan(chopb_mean_frame)] = 0
    chopa_mean_frame[np.isnan(chopa_mean_frame)] = 0

    wx = np.linspace(0, 2*windowsize-1, 2*windowsize)
    wy = np.linspace(0, 2*windowsize-1, 2*windowsize)
    wx, wy = np.meshgrid(wx, wy)
    
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

    return fwhms, eccentricities, np.asarray(psfmaxima), np.asarray(background_dev), np.asarray(correlations), np.asarray(amplitudes), np.asarray(gauss_offsets)

def frame_rejection(psfmaxima, background_dev, fwhms, eccentricities, correlations, amplitudes, gauss_offsets, sigma=1.5):
        
    return (background_dev < np.nanmedian(background_dev) + sigma*np.nanstd(background_dev)) &\
        (psfmaxima > np.nanmedian(psfmaxima) - sigma*np.nanstd(psfmaxima)) &\
        (correlations > np.nanmedian(correlations) - sigma*np.nanstd(correlations)) &\
        (amplitudes > np.nanmedian(amplitudes) - sigma*np.nanstd(amplitudes)) &\
        (fwhms < np.nanmedian(fwhms) + sigma*np.nanstd(fwhms)) &\
        (eccentricities < np.nanmedian(eccentricities) + sigma*np.nanstd(eccentricities)) &\
        (gauss_offsets < np.nanmedian(gauss_offsets) + sigma*np.nanstd(gauss_offsets))

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

def frame_binning(aligned_files, raw_dir, frame_bool, chops, para_angles, array_shape, bin=50, threadcount=500):
    
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
    
    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        a_binned_angles, a_binned_filenames  = zip(*tqdm(pool.imap(BinFrames((array_shape, binned_dir)), zip(a_angles, a_binfiles))))
    a_binned_angles, a_binned_filenames = np.asarray(a_binned_angles), np.asarray(a_binned_filenames)
    
    #if __name__ == "__main__":
    with Pool(threadcount) as pool:
        b_binned_angles, b_binned_filenames = zip(*tqdm(pool.imap(BinFrames((array_shape, binned_dir)), zip(b_angles, b_binfiles))))
    b_binned_angles, b_binned_filenames = np.asarray(b_binned_angles), np.asarray(b_binned_filenames)
  
    binned_files = np.asarray(list(pathlib.Path(str(binned_dir)).rglob('*.fits')))
    binned_filenames = np.asarray([f.name for f in binned_files])

    binned_chops = np.empty(len(binned_files), dtype="<U16")
    binned_angles = np.zeros(len(binned_files))
    
    binned_chops[np.where(np.isin(binned_filenames,a_binned_filenames) == True)[0]] = "CHOP_A"
    binned_chops[np.where(np.isin(binned_filenames,a_binned_filenames) == True)[0]] = "CHOP_B"

    for i in range(len(binned_files)):
        if binned_filenames[i] in a_binned_filenames:
            binned_angles[i] = a_binned_angles[np.where(a_binned_filenames == binned_filenames[i])[0]]
        else:
            binned_angles[i] = b_binned_angles[np.where(b_binned_filenames == binned_filenames[i])[0]]
    
    return binned_files, binned_chops, binned_angles
    
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

def circular_mask(center, radius, width, height):
    '''
    Creates circular mask of certain radius in an image of certain width and height
    '''
    Y, X = np.ogrid[:height, :width]
    distance = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = distance <= radius
    
    return mask

def pad_frame(frame, new_xlen, new_ylen, padding_x, padding_y):
    
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