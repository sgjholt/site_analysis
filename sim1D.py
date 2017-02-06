import sys
sys.path.insert(0, '../SeismicSiteTool/')
import site_class as sc
import libs.SiteModel as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from utils import binning, pick_model, define_model_space, silent_remove, df_cols


class Sim1D(sc.Site, sm.Site1D):
    model_space = []
    run_dir = ''
    simulation_path = ''
    def __init__(self, name, working_directory, run_dir=None, litho=False, vel_file_dir=None):

        sc.Site.__init__(self, name, working_directory, vel_file_dir)
        sm.Site1D.__init__(self)
        self.litho = litho
        self.__add_site_profile()
        if run_dir is not None:
            self.run_dir = run_dir

    def __add_site_profile(self):

        if not self.has_vel_profile: # printing is handled in the Site class to warn the user - just return None
            return None

        vels = self.get_velocity_profile(self.litho)
        for i, hl in enumerate(vels['thickness']):
            self.AddLayer([hl, vels['vp'][i], vels['vs'][i], vels['rho'][i], 10, 10])
        if self.litho:  # add final half layer
            self.AddLayer([0, vels['vp'][-1], vels['vs'][-1], vels['rho'][-1], 10, 10])

    def elastic_forward_model(self, i_ang=0, elastic=True, plot_on=False, show=False, freqs=None):

        if not self.has_vel_profile:
            print('Cannot model - no velocity model')
            return None

        if freqs is None:
            self.GenFreqAx(0.1, 25, 1000)
        else:
            self.GenFreqAx(freqs[0], freqs[1], freqs[2])

        shtf = self.ComputeSHTF(i_ang, elastic)

        if self.litho:
            model = 'lithology'
        else:
            model = 'standard'

        if elastic:
            types = 'Elastic'
        else:
            types = 'Anelastic'

        if plot_on:
            plt.title('{0} : 1D Linear SHTF'.format(self.site))
            plt.loglog(self.Amp['Freq'], np.abs(shtf), label='SHTF: {0} - {1}'.format(types, model))
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

        observed = self.sb_ratio().loc['mean'].values  # pandas DataFrame object

        predicted = self.elastic_forward_model(i_ang, elastic, freqs=freqs)

        if predicted is None:  # No forward model - return nothing
            print('Misfit not available - no forward model.')
            return None  # return nothing to break out of function

        log_residuals = np.log(predicted.reshape(1, len(predicted))[0]) - observed

        log_rms_misfit = (np.sum(log_residuals ** 2) / len(log_residuals)) ** 0.5

        bin_log_resids, bin_freqs = binning(log_residuals, self.Amp['Freq'], 10)

        if plot_on:
            plt.title('{0} : Log Residuals - Log RMS Misfit: {1}.'.format(self.site, round(log_rms_misfit), 2))
            plt.plot(bin_freqs, bin_log_resids, 'ko', label='Residuals')
            plt.hlines(0, 0.1, 25, linestyles='dashed', colors='red')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Log Residuals')
        else:
            return log_residuals, log_rms_misfit
        if show:
            plt.show()

    def random_search(self, pct_variation, steps, iterations, name, elastic=False):

        if not name.endswith('.txt'):  # in case you forget to add it
            name.join('.txt')

        self.simulation_path = self.run_dir + name

        self.model_space = define_model_space(self.Mod['Vs'], pct_variation, steps)  # build the model space

        dimensions, indexes = self.model_space.shape  # log the dimensions of the model space

        random_choices = np.random.randint(0, indexes-1, (iterations, dimensions))  # pick indices at random
                                                                                    # (from uniform distribution) and
                                                                                    # build realisations from the model space
        realisations = np.zeros((iterations, dimensions))  # initialise empty numpy array to be populated
        for i, row in enumerate(random_choices):
            realisations[i] = pick_model(self.model_space, row)  # pick model from model space using indexes


        results = pd.DataFrame(columns=df_cols(dimensions=dimensions))  # build pd DataFrame to store results
        results.index.name = 'iteration'

        # run original model
        _, rms = self.misfit(elastic=elastic)
        results.loc[0] = self.Amp['Vs'].tolist() + [10] + [rms]

        for i, model in enumerate(realisations):
            self.
            results.loc[i+1] = model.









        # perms = itertools.product([x for x in range(indexes)], repeat=dimensions)

        # for i, perm in enumerate(perms):
        #    model = pick_model(model_space, perm)
        return None

