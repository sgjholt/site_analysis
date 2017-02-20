import sys
sys.path.insert(0, '../SeismicSiteTool/')
import site_class as sc
import libs.SiteModel as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from utils import binning, pick_model, define_model_space, silent_remove, df_cols, calc_density_profile, exp_cdf


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
        if self.litho:
            self.vp_vs = np.array(self.Mod['Vp']) / np.array(self.Mod['Vs'])

    def __add_site_profile(self):

        if not self.has_vel_profile: # printing is handled in the Site class to warn the user - just return None
            return None

        vels = self.get_velocity_profile(litho=self.litho)
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

        self.ComputeSHTF(i_ang, elastic)

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
            plt.loglog(self.Amp['Freq'], np.abs(self.Amp['Shtf']), label='SHTF: {0} - {1}'.format(types, model))
            plt.hlines(1, 0.1, 25, linestyles='dashed', colors='red')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('SHTF')

        else:
            return np.abs(self.Amp['Shtf'])
        if show:
            plt.show()

    def misfit(self, weights=(0.4, 0.6), lam=1, i_ang=0, elastic=True, plot_on=False, show=False, cadet_correct=False):

        sb_table = self.sb_ratio(cadet_correct=cadet_correct)

        freqs = (round(float(
            sb_table.columns.values[0]), 2), round(float(
                sb_table.columns.values[-1]), 2), len(sb_table.columns.values))

        observed = sb_table.loc['mean'].values   # observed (mean of ln values) (normally distributed in logspace)
        std = sb_table.loc['std'].values  # std of ln values

        predicted = self.elastic_forward_model(i_ang, elastic, freqs=freqs)

        if predicted is None:  # No forward model - return nothing
            print('Misfit not available - no forward model.')
            return None  # return nothing to break out of function

        log_residuals = (np.log(predicted.reshape(1, len(predicted))[0]) - observed)/std  # weighted by stdv
        log_residuals /= np.abs(log_residuals).max()  # normalise between 0-1

        log_rms_misfit = (np.sum(log_residuals ** 2) / len(log_residuals)) ** 0.5  # amplitude quality of fit

        x_cor = np.correlate(
            observed, np.log(predicted.reshape(1, len(predicted))[0]), 'full').argmax() - (
                np.correlate(observed, np.log(predicted.reshape(1, len(predicted))[0]), 'full').__len__()-1)/2

        total_misfit = log_rms_misfit*weights[0] + exp_cdf(np.abs(x_cor)/0.01, lam=lam)*weights[1]

        bin_log_resids, bin_freqs = binning(log_residuals, self.Amp['Freq'], 10)

        if plot_on:
            plt.title('{0} : Log Residuals - Log RMS Misfit: {1}.'.format(self.site, round(log_rms_misfit), 2))
            plt.semilogx(bin_freqs, bin_log_resids, 'ko', label='Residuals')
            plt.hlines(0, 0.1, 25, linestyles='dashed', colors='red')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Log Residuals')
        else:
            return log_residuals, log_rms_misfit, x_cor, total_misfit
        if show:
            plt.show()

    def uniform_random_search(self, pct_variation, steps, iterations, name, elastic=False):
        # UNFINISHED
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
        results.loc[0] = self.Mod['Vs'].tolist() + [10] + [rms]

        for i, model in enumerate(realisations):
            self.__modify_site_model(model)
            self.misfit(elastic=elastic)
            results.loc[i + 1] = model.tolist() + [rms]

    def modify_site_model(self, model):
        """
        This function will modify the site model based on a specified Vs profile plus Qs value (model[-1])

        :param model: np.ndarray object containing Vs values in m/s model[:-1] and model[-1] is a Qs value
        :return: None - this is an inplace method
        """
        for j, var in enumerate(model):  # assign vs values given in model to site
            if j != len(model)-1:
                self.Mod['Vs'][j] = var
                self.Mod['Qs'][j] = model[-1]

        vp = self.Mod['Vs'] * self.vp_vs   # use vp/vs to calculate Vp values such that physical properties are
                                           # consistent in each layer
        dn = calc_density_profile(np.array(self.Mod['Vp']) / 1000) * 1000  # calculate density based on Vp
        for j, var in enumerate(vp):  # assign values of vp and density to site
            self.Mod['Vp'][j] = var  # vp values
            self.Mod['Dn'][j] = dn[j]  # density values

        if self.litho:  # extra layer to consider - added half layer at base
            for key in ['Vs', 'Vp', 'Dn']:
                self.Mod[key][-1] = self.Mod[key][-2]  # make sure half layer = layer above (Thompson, 2012)











        # perms = itertools.product([x for x in range(indexes)], repeat=dimensions)

        # for i, perm in enumerate(perms):
        #    model = pick_model(model_space, perm)
        return None


