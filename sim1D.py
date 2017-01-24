import sys
sys.path.insert(0, '../SeismicSiteTool/')
import site_class as sc
import libs.SiteModel as sm
import matplotlib.pyplot as plt
import numpy as np


class Sim1D(sc.Site, sm.Site1D):
    def __init__(self, name, working_directory, vel_file_dir=None):

        sc.Site.__init__(self, name, working_directory, vel_file_dir)
        sm.Site1D.__init__(self)
        self.__add_site_profile()

    def __add_site_profile(self):
        if not self.has_vel_profile:
            return None

        vels = self.get_velocity_profile()
        for i, hl in enumerate(vels['thickness']):
            self.AddLayer([hl, vels['vp'][i], vels['vs'][i], vels['rho'][i], 1, 1])

    def elastic_forward_model(self, i_ang=0, elastic=True, plot_on=False, show=False, freqs=None):

        if not self.has_vel_profile:
            print('Cannot model - no velocity model')
            return None

        if freqs is None:
            self.GenFreqAx(0.1, 25, 1000)
        else:
            self.GenFreqAx(freqs[0], freqs[1], freqs[2])

        shtf = self.ComputeSHTF(i_ang, elastic)

        if plot_on:
            plt.title('{0} : 1D Linear Elastic SHTF'.format(self.site))
            plt.loglog(self.Freq, np.abs(shtf), label='SHTF')
            plt.hlines(1, 0.1, 25, linestyles='dashed', colors='red')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('SHTF')

        else:
            return np.abs(shtf)
        if show:
            plt.show()

    def misfit(self, i_ang=0, elastic=True, plot_on=False, show=False):

        freqs = (round(float(
            self.sb_ratio().columns.values[0]), 2), round(float(
            self.sb_ratio().columns.values[-1]), 2), len(self.sb_ratio().columns.values))

        observed = self.sb_ratio().loc['mean']  # pandas DataFrame object

        predicted = self.elastic_forward_model(i_ang, elastic, freqs=freqs)

        if predicted is None:  # No forward model - return nothing
            print('Misfit not available - no forward model.')
            return None  # return nothing to break out of function

        log_residuals = np.log(predicted) - observed

        log_rms_misfit = (np.sum(log_residuals ** 2) / len(log_residuals)) ** 0.5

        if plot_on:
            plt.title('{0} : Log Residuals - Log RMS Misfit: {1}.'.format(self.site, round(log_rms_misfit), 2))
            plt.loglog(self.Freq, log_residuals, 'ko', label='Residuals')
            plt.hlines(0, 0.1, 25, linestyles='dashed', colors='red')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Log Residuals')
        else:
            return log_residuals, log_rms_misfit
        if show:
            plt.show()
