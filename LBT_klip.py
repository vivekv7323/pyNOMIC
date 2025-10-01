import pyklip.parallelized as parallelized
from pyklip.instruments.Instrument import Data
import numpy as np
from astropy.io import fits

class LBT(Data):
    """
    This is my new instrument class
    """
    ####################
    ### Constructors ###
    ####################
    def __init__(self, filepaths=None, PAs=None, framew=512, frameh=512, numthreads=-1, origin=None):
        """
        Initialization code for GPIData

        Note:
            Argument information is in the GPIData class definition docstring
        """
        super(LBT, self).__init__()

        if filepaths is None:
            self._input == None
            self._filenames = None
            self._filenums = None
            self._wvs = None
            self._PAs = None
            self._wcs = None
        else:
            self.readdata(filepaths, PAs=PAs, framew=framew, frameh=frameh)
        self._output = None
        self.output_centers = None
        self._IWA = 20
        if origin is None:
            self._centers = None
        else:
            self._centers = np.asarray([(origin[0], origin[1])]*len(filepaths))
        
    ################################
    ### Instance Required Fields ###
    ################################
    @property
    def input(self):
        return self._input
    @input.setter
    def input(self, newval):
        self._input = newval

    @property
    def centers(self):
        return self._centers
    @centers.setter
    def centers(self, newval):
        self._centers = newval

    @property
    def filenums(self):
        return self._filenums
    @filenums.setter
    def filenums(self, newval):
        self._filenums = newval

    @property
    def filenames(self):
        return self._filenames
    @filenames.setter
    def filenames(self, newval):
        self._filenames = newval

    @property
    def PAs(self):
        return self._PAs
    @PAs.setter
    def PAs(self, newval):
        self._PAs = newval

    @property
    def wvs(self):
        return self._wvs
    @wvs.setter
    def wvs(self, newval):
        self._wvs = newval

    @property
    def wcs(self):
        return self._wcs
    @wcs.setter
    def wcs(self, newval):
        self._wcs = newval

    @property
    def IWA(self):
        return self._IWA
    @IWA.setter
    def IWA(self, newval):
        self._IWA = newval

    @property
    def output(self):
        return self._output
    @output.setter
    def output(self, newval):
        self._output = newval

    def readdata(self, filepaths, PAs=None, framew=512, frameh=512):
        self._input = np.zeros((len(filepaths), framew, frameh))
        self._filenames = np.empty(len(filepaths), dtype='<U25')
        self._filenums = np.linspace(0, len(filepaths)-1, len(filepaths))
        self._wvs = np.ones(len(filepaths))
        if PAs is None:
            self._PAs = np.zeros(len(filepaths))
        else:
            self._PAs = PAs
        self._wcs = [None]*len(filepaths)
        for i in range(len(filepaths)):
            hdul = fits.open(filepaths[i])
            frame = hdul[0].data
            #frame[np.isnan(frame)] = 0
            self._input[i] = frame
            self._filenames[i] = filepaths[i].name
            hdul.close()

    def savedata(self, filepath, data, klipparams=None, filetype="", zaxis=None, more_keywords=None):
           newhdul = fits.HDUList([fits.PrimaryHDU(data=data)])
           newhdul.writeto(filepath, overwrite=True)
           newhdul.close() 
