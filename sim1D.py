import sys
sys.path.insert(0, '../SeismicSiteTool/')
import site_class as sc
import libs.SiteModel as sm
import matplotlib.pyplot as plt
import numpy as np


class Sim1D(sc.Site, sm.Site1D, freqs=None):
    def __init__(self, name, working_directory, freqs, vel_file_dir=None):
        sc.Site.__init__(self, name, working_directory, vel_file_dir)
        sm.Site1D.__init__(self)
        self.__add_site_profile()
        if freqs is None:
            self.GenFreqAxis(0.1, 25, 1000)
        else:
            self.GenFreqAxis(freqs[0], freqs[1], freqs[3])

    def __add_site_profile(self):
        vels = self.get_velocity_profile()
        for i, hl in enumerate(vels['thickness']):
            self.AddLayer([hl, vels['vp'][i], vels['vs'][i], vels['rho'][i], 1, 1])

    def forward_model(self, i_ang=0, elastic=True, plot_on=False, show=False):

        shtf = self.ComputeSHTF(i_ang, elastic)

        if plot_on:
            plt.loglog(self.Freqs, np.abs(shtf), label='SHTF')

        else:
            return np.abs(shtf)
        if show:
            plt.show()
