# built-ins
import sys
import glob
import math
sys.path.insert(0, '../SeismicSiteTool/')
# 3rd party dependencies
import numpy as np
import pandas as pd
import scipy.stats as stats
from obspy.signal.konnoohmachismoothing import konno_ohmachi_smoothing
# my code / custom libs (SiteModel)
import site_class as sc
import libs.SiteModel as sm
import matplotlib.pyplot as plt
from utils import pick_model, uniform_model_space, df_cols, calc_density_profile, fill_troughs, \
    uniform_sub_model_space, sig_resample
from parsers import parse_simulation_cfg
# import itertools


class Sim1D(sc.Site, sm.Site1D):

    model_space = []
    run_dir = ''
    simulation_path = ''
    sim_pars = {}
    sim_results = []
    VLER = False

    def __init__(self, name, working_directory, run_dir=None, litho=False, vel_file_dir=None):
        """

        :param name:
        :param working_directory:
        :param run_dir:
        :param litho:
        :param vel_file_dir:q
        """
        sc.Site.__init__(self, name, working_directory, vel_file_dir)
        sm.Site1D.__init__(self)
        self.litho = litho
        self.__add_site_profile()
        if run_dir is not None:
            self.run_dir = run_dir
        if self.litho:
            self.vp_vs = np.array(self.Mod['Vp']) / np.array(self.Mod['Vs'])
        if name[:-2] == 'VLER':
            self.VLER = True

    def __add_site_profile(self, custom_model=None, q_model=False):
        """

        :return:
        """
        # hard reset the model parameters to avoid background problems
        # self.Mod = []

        if not self.has_vel_profile:  # printing is handled in the Site class to warn the user - just return None
            return None

        if custom_model is None:  # use the kik-net supplied model
            vels = self.get_velocity_profile(litho=self.litho)
            for i, hl in enumerate(vels['thickness']):
                self.AddLayer([hl, vels['vp'][i], vels['vs'][i], vels['rho'][i], 100, 100])
            if self.litho:  # add final half layer
                self.AddLayer([0, vels['vp'][-1], vels['vs'][-1], vels['rho'][-1], 100, 100])
            if q_model:
                self.modify_site_model(model=np.array(self.vel_profile['vs'].tolist() + [None]), q_model=q_model)
        else:
            self.modify_site_model(model=custom_model, q_model=q_model)

    def linear_forward_model_1d(self, i_ang=0, elastic=True, konno_ohmachi=None, motion='outcrop', log_sample=None,
                                plot_on=False, show=False):
        """

        :param i_ang:
        :param elastic:
        :param konno_ohmachi:
        :param motion:
        :param log_sample: tuple (no. samples, log_base), else: None : log sampling of predicted and observed
        :param plot_on:
        :param show:
        :return:
        """
        if not self.has_vel_profile:
            print('Cannot model - no velocity model')
            return None

        if len(self.Amp['Freq']) == 0:
            if log_sample is not None:
                self.Amp['Freq'] = np.logspace(math.log(0.1, log_sample[1]), math.log(25, log_sample[1]),
                                               log_sample[0], base=log_sample[1])
            else:
                self.Amp['Freq'] = self.sb_ratio().columns.values.astype(float).tolist()
        else:
            if log_sample is not None and len(self.Amp['Freq']) != log_sample[1]:
                self.Amp['Freq'] = np.logspace(math.log(0.1, log_sample[1]), math.log(25, log_sample[1]),
                                               log_sample[0], base=log_sample[1])
            elif log_sample is None and len(self.Amp['Freq']) != len(self.sb_ratio().columns.values):
                self.Amp['Freq'] = self.sb_ratio().columns.values.astype(float).tolist()
            else:
                pass

        self.ComputeSHTF(i_ang, elastic, motion)

        if konno_ohmachi is not None:
            # if not none then value should be int containing bandwidth parameter b
            self.Amp['Shtf'] = konno_ohmachi_smoothing(np.abs(self.Amp['Shtf'])[::, 0],
                                                       np.array(self.Amp['Freq']).astype(float),
                                                       bandwidth=konno_ohmachi, normalize=True)

        if plot_on:
            if self.litho:
                model = 'lithology'
            else:
                model = 'standard'

            if elastic:
                types = 'Elastic'
            else:
                types = 'Anelastic'
            plt.title('{0} : 1D Linear SHTF'.format(self.site))
            plt.loglog(self.Amp['Freq'], np.abs(self.Amp['Shtf']), label='SHTF: {0} - {1}'.format(types, model))
            plt.hlines(1, 0.1, 25, linestyles='dashed', colors='red')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('SHTF')
            if show:
                plt.show()

        else:
            if konno_ohmachi is not None:
                return np.abs(self.Amp['Shtf'])
            else:
                return np.abs(self.Amp['Shtf'])[::, 0]

    def misfit(self, i_ang=0, x_cor_range=(0, 25), elastic=True, motion='outcrop', plot_on=False, show=False,
               cadet_correct=False, fill_troughs_pct=None, konno_ohmachi=None, log_sample=None):
        """

        
        :param i_ang: incident angle (radians) of up-going wave from bedrock for forward model - int/float
        :param x_cor_range: range of values used for x_correlation - take values past unity part of response
                               (ratio ~= 1) to avoid -ve bias in correlation. - tuple (int/float, int/float)
        :param elastic: perform elastic or anelastic simulation - bool - True/False
        :param motion: Choose to model 'within' (A+B) or 'outcrop' (2*A) motion at depth - str
        :param plot_on: plot waveform in matplotlib (background) - bool - True/False
        :param show: show matplotlib plot - bool- True/False
        :param cadet_correct: Apply the correction to SB ratio detailed in Cadet et al. (2012) - bool - True/False
        :param fill_troughs_pct: Fill the troughs of theoretical waveform to a percentage of peak amplitudes
        :param konno_ohmachi
        :param log_sample: tuple (no. samples, log_base), else: None
        :return: None: if plot_on == True: else: tuple (log_resids, log_misfit, x_cor):
        """

        # get the pandas table for sb site sb ratio
        predicted = self.linear_forward_model_1d(i_ang, elastic, motion=motion, konno_ohmachi=konno_ohmachi,
                                                 log_sample=log_sample)
        if predicted is None:  # No forward model - return nothing
            print('Misfit not available - no forward model.')
            return None  # return nothing to break out of function
        if fill_troughs_pct is not None:
            predicted = fill_troughs(predicted, fill_troughs_pct)
        observed = self.sb_ratio(cadet_correct=cadet_correct).loc[
            'mean'].values  # observed (mean of ln values) (normally distributed in log_space)
        # std = sb_table.loc['std'].values  # std of ln values
        _freqs = np.array(self.Amp['Freq'])  # str by default
        # freqs = (round(float(_freqs[0]), 2), round(float(_freqs[-1]), 2), len(_freqs))  # specify freqs for fwd model
        # calc fwd model
        if log_sample is not None:  # must re-sample observed signal to match theoretical
            observed = sig_resample(_freqs, observed, self.sb_ratio().columns.values.astype(float))

        log_residuals = (np.log(predicted) - observed)  # /std  # weighted by stdv

        # if log_residuals.min() < 0:  # if lowest is negative need to apply linear shift to 0
        #     log_residuals -= log_residuals.min()
        # if np.abs(log_residuals).max() < 1:
        #     log_residuals /= 1
        # else:
        #     log_residuals /= np.abs(log_residuals).max()  # normalise between -1 to 1

        log_rms_misfit = (np.sum(log_residuals ** 2) / len(log_residuals)) ** 0.5  # amplitude GOF between 0-1
        # re-sample the predicted and observed signals to the range specified for x_correlation
        x_cor_p = np.log(predicted[(_freqs >= x_cor_range[0]) & (_freqs <= x_cor_range[1])])
        x_cor_o = observed[(_freqs >= x_cor_range[0]) & (_freqs <= x_cor_range[1])]
        # Perform the x_correlation - take arg max and subtract half the total length to get the 'frequency lag'
        x_cor = np.correlate(x_cor_o / np.mean(x_cor_o), x_cor_p / np.mean(x_cor_p),
                             'full')  # do x_corr, store in memory - efficient for large sims
        max_xcor = x_cor.max()  # max value

        if log_sample is None:
            if self.VLER is False:
                dt = 0.01  # time delta - ***TEMPORARY - NEEDS TO BE MORE FLEXIBLE ***
                x_cor = (x_cor.argmax() - (
                len(x_cor) - 1) / 2) * dt  # len -1 because the signal index begins counting at 0
            else:
                log_space_dt = np.log(_freqs).max() / len(np.log(_freqs))
                ln_f_lag = (np.linspace(np.log(1), len(x_cor), len(x_cor)) * log_space_dt - np.log(_freqs.max()))
                x_cor = ln_f_lag[x_cor.argmax()]  # calculate lag using log_f since signal is log-sampled
        else:
            log_space_dt = np.log(_freqs).max() / len(np.log(_freqs))
            ln_f_lag = (np.linspace(np.log(1), len(x_cor), len(x_cor)) * log_space_dt - np.log(_freqs.max()))
            x_cor = ln_f_lag[x_cor.argmax()]  # calculate lag using log_f since signal is log-sampled

        # Calculate total misfit in both amplitude and frequency (fitting in both dimensions)
        # ***NO LONGER CALCULATED***
        # total_misfit = log_rms_misfit*weights[0] + exp_cdf(np.abs(x_cor), lam=lam)*weights[1]

        if plot_on:
            plt.figure(figsize=(10, 10))
            plt.subplot(211)
            # bin_log_resids, bin_freqs = binning(log_residuals, self.Amp['Freq'], 10)  # bin only when plotting
            plt.title('{0} : Ln Residuals - Misfit: ({1}, {2}).'.format(self.site, np.round(log_rms_misfit, 5),
                                                                        np.round(x_cor, 5)))
            plt.plot(_freqs, log_residuals, 'ko', label='Residuals')
            plt.hlines(0, 0.1, 25, linestyles='dashed', colors='red')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Ln Residuals')
            plt.grid(which='both')
            plt.legend()

            plt.subplot(212)
            plt.plot(_freqs, observed, 'k', label='ln(observed)')
            plt.plot(_freqs, np.log(predicted), 'b', label='ln(predicted)')
            plt.hlines(0, 0.1, 25, linestyles='dashed', colors='red')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Ln(SB Ratio) / Ln(SHTF)')
            plt.grid(which='both')
            plt.legend()
            plt.show()
            # self.linear_forward_model_1d(elastic=elastic, plot_on=True, freqs=freqs)
            # self.plot_sb(stdv=(1,))

        else:
            return log_residuals, log_rms_misfit, x_cor, max_xcor
        if show:
            plt.show()

    def modify_site_model(self, model, q_model=False):
        """
        This function will modify the site model based on a specified Vs profile plus Qs value (model[-1])

        :param model: np.ndarray object containing Vs values in m/s model[:-1] and model[-1] is a Qs value
        :param sub_layers: does the model need to adapt to a number of sub-layers?
        :return: None - this is an inplace method
        """
        # if sub_layers:  # need to modify the whole model to account for addition of sub-layers
        #  self.Mod = {'Dn': [], 'Hl': [], 'Qp': [], 'Qs': [], 'Vp': [], 'Vs': []}
        #  case 0 - 'Hl' has not been changed to represent sublayer thicknesses - not the correct length
        #  must change Thicknesses, Density, Vp, Vs, Qp an Qs
        if len(self.vel_profile['thickness']) != (len(model) - 1):  # if it has sub-layering - calc new layer thicks
            print('Calculating new layer thicknesses for given sub-layers')
            subl_factor = int((len(model) - 1) / (len(self.vel_profile['thickness']) - 1))
            # print(subl_factor)  # how many sub-layers were used
            # # hl = np.zeros(len(model)-1)
            hl = []
            # # vp_vs = np.zeros(len(model)-1)
            vp_vs = []
            for i, zipped in enumerate(zip(self.vel_profile['thickness'], self.vp_vs)):
                if i < len(self.vel_profile['thickness']) - 1:
                    Hl, Vp_Vs = zipped
                    for n in range(subl_factor):
                        hl.append(Hl / subl_factor)
                        vp_vs.append(Vp_Vs)
            hl.append(self.vel_profile['thickness'][-1])  # add the half space layer
            vp_vs.append(self.vp_vs[-1])  # vp/vs ratio for half space layer
            # print(len(vp_vs), len(hl))
            self.vp_vs = np.array(vp_vs)  # assign vp_vs back to self
            self.Mod['Hl'] = hl  # assign layer thicknesses back to self
        else:  # if there are no sub-layers
            self.Mod['Hl'] = self.vel_profile['thickness']

        self.Mod['Vs'] = model[:-1].tolist()
        self.Mod['Vp'] = (self.Mod['Vs'] * self.vp_vs).tolist()
        self.Mod['Dn'] = (calc_density_profile(np.array(self.Mod['Vp']) / 1000) * 1000).tolist()
        if q_model:
            self.Mod['Qs'] = (((np.array(self.Mod['Vs']) / 100) ** 1.6) + 10).tolist()
        else:
            self.Mod['Qs'] = [model[-1] for _ in range(model[:-1].size)]
        self.Mod['Qp'] = self.Mod['Qs']
        # print(self.Mod)
        return None

        # else:
        #    for j, var in enumerate(model):  # assign vs values given in model to site
        #        if j != len(model) - 1:
        #            self.Mod['Vs'][j] = var
        #            self.Mod['Qs'][j] = model[-1]

        #    vp = self.Mod['Vs'] * self.vp_vs   # use vp/vs to calculate Vp values such that physical properties are
        #                                       # consistent in each layer
        #    dn = calc_density_profile(np.array(self.Mod['Vp']) / 1000) * 1000  # calculate density based on Vp
        #    for j, var in enumerate(vp):  # assign values of vp and density to site
        #        self.Mod['Vp'][j] = var  # vp values
        #        self.Mod['Dn'][j] = dn[j]  # density values

        #   if self.litho:  # extra layer to consider - added half layer at base
        #       for key in ['Vs', 'Vp', 'Dn']:
        #            self.Mod[key][-1] = self.Mod[key][-2]  # make sure half layer = layer above (Thompson, 2012)

            # perms = itertools.product([x for x in range(indexes)], repeat=dimensions)

            # for i, perm in enumerate(perms):
            #    model = pick_model(model_space, perm)
            # print(self.Mod)
        #    return None

    def uniform_sub_random_search(self, pct_variation, steps, iterations, name, i_ang=0,
                                  x_cor_range=(0, 25), const_q=None, n_sub_layers=(), elastic=True, cadet_correct=False,
                                  fill_troughs_pct=None, save=False, gaussian_sampling=True, debug=False,
                                  motion='outcrop', konno_ohmachi=None, log_sample=None):
        """
        UNFINISHED

        :param pct_variation: percentage about the original vs model value to vary
        :param steps: amount of steps - more steps = less space between Vs values
        :param iterations: number of model space realisations - chosen at random
        :param name: the name of the simulation (file to be saved)
        :param weights: weights for misfit function - see misfit method
        :param lam: lambda for exponential CDF - see misfit method
        :param i_ang: incident angle of upgoing wave - rads
        :param x_cor_range: range for x_correlation - see misfit method
        :param const_q: if not None provide value for constant damping
        :param n_sub_layers
        :param elastic: elastic or anelastic simulation
        :param cadet_correct: apply Cadet et al. 2012 correction to observed SB ratio
        :param fill_troughs_pct:
        :param save: save the result as csv file
        :param gaussian_sampling
        :param debug: bool: If true perform debugging actions. 
        :param motion:
        :param konno_ohmachi:
        :param log_sample:
        :return:
        """
        # -------------------------------------run 0-----------------------------------------------------#
        # SETUP START #
        if const_q is not None:
            self.Mod['Qs'] = [const_q for _ in self.Mod['Vs']]
        else:
            self.reset_site_model(q_model=True)

        self.simulation_path = self.run_dir + name

        self.model_space = uniform_model_space(self.Mod['Vs'], pct_variation, steps,
                                               const_q=const_q)  # build the model space

        dimensions, indexes = self.model_space.shape  # log the dimensions of the model space

        if gaussian_sampling:
            lower, upper = 0, indexes - 1
            mu, sigma = (indexes - 1) / 2, indexes * 0.2
            pdf = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
            random_choices = np.round(pdf.rvs((iterations, dimensions)), 0).astype(int)

        else:
            random_choices = np.random.randint(0, indexes - 1, (iterations, dimensions))  # pick indices at random
        # (from uniform distribution) and
        # build realisations from the model space
        realisations = np.zeros((iterations, dimensions))  # initialise empty numpy array to be populated
        for i, row in enumerate(random_choices):
            realisations[i] = pick_model(self.model_space, row)  # pick model from model space using indexes

        # store results to return back to me - otherwise just using unnecessary cpu/memory
        all_results = []

        results = pd.DataFrame(
            columns=df_cols(dimensions=dimensions, sub_layers=True))  # build pd DataFrame to store results
        results.index.name = 'trial'

        # SETUP END #

        # run original model
        _, amp_mis, freq_mis, max_xcor = self.misfit(elastic=elastic, cadet_correct=cadet_correct,
                                                     fill_troughs_pct=fill_troughs_pct,
                                                     i_ang=i_ang, x_cor_range=x_cor_range, motion=motion,
                                                     konno_ohmachi=konno_ohmachi, log_sample=log_sample)
        if debug:
            self.misfit(elastic=elastic, cadet_correct=cadet_correct,
                        fill_troughs_pct=fill_troughs_pct, i_ang=i_ang, x_cor_range=x_cor_range, plot_on=True,
                        motion=motion, konno_ohmachi=konno_ohmachi, log_sample=log_sample)
        print(
            "Trial:{0}-Model:{1}-Misfit:({2},{4})-N_sub_layers:{3}".format(0, self.Mod['Vs'] + [self.Mod['Qs'][0]],
                                                                           np.round(amp_mis, 3), 0,
                                                                           np.round(freq_mis, 3)))
        # store result in pandas data frame
        results.loc[0] = self.Mod['Vs'] + [self.Mod['Qs'][0]] + [amp_mis, freq_mis, max_xcor, 0]
        # loop over the model realisations picked at random and calculate misfit
        for i, model in enumerate(realisations):
            if const_q is None:  # i.e. we're using a qs/vs relation
                self.modify_site_model(model, q_model=True)
            else:
                self.modify_site_model(model)
            # calculate misfit
            _, amp_mis, freq_mis, max_xcor = self.misfit(elastic=elastic, cadet_correct=cadet_correct,
                                                         fill_troughs_pct=fill_troughs_pct,
                                                         i_ang=i_ang, x_cor_range=x_cor_range, motion=motion,
                                                         konno_ohmachi=konno_ohmachi, log_sample=log_sample)
            if debug:
                self.misfit(elastic=elastic, cadet_correct=cadet_correct,
                            fill_troughs_pct=fill_troughs_pct, i_ang=i_ang, x_cor_range=x_cor_range, plot_on=True,
                            motion=motion, konno_ohmachi=konno_ohmachi, log_sample=log_sample)
            # store result in data frame
            results.loc[i + 1] = model.tolist() + [amp_mis, freq_mis, max_xcor, 0]
            print("Trial:{0}-Model:{1}-Misfit:({2},{4})-N_sub_layers:{3}".format(i + 1, model, np.round(amp_mis, 3), 0,
                                                                                 np.round(freq_mis, 3)))
        if save:
            results.to_csv(self.simulation_path + 'n_sub_' + str(0) + '.csv')
        else:
            all_results.append(results)
        # -------------------------------------run sub-layers-----------------------------------------------------#

        for n_layers in n_sub_layers:  # loop over sub-layer trials
            # SETUP START #
            self.reset_site_model()
            if const_q is not None:
                self.Mod['Qs'] = [const_q for _ in self.Mod['Vs']]
            # build the model space
            model_space, orig_sub = uniform_sub_model_space(self.Mod['Vs'], variation_pct=pct_variation, steps=steps,
                                                            n_sub_layers=n_layers, const_q=const_q)

            dimensions, indexes = model_space.shape  # log the dimensions of the model space

            if gaussian_sampling:  # sample indexes in accordance to truncated Gaussian distribution
                lower, upper = 0, indexes - 1
                mu, sigma = (indexes - 1) / 2, indexes * 0.2
                pdf = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
                random_choices = np.round(pdf.rvs((iterations, dimensions)), 0).astype(int)

            else:
                random_choices = np.random.randint(0, indexes - 1, (iterations, dimensions))  # pick indices at random
            # (from uniform distribution) and
            # build realisations from the model space
            realisations = np.zeros((iterations, dimensions))  # initialise empty numpy array to be populated
            for i, row in enumerate(random_choices):
                realisations[i] = pick_model(model_space, row)  # pick model from model space using indexes

            results = pd.DataFrame(
                columns=df_cols(dimensions=dimensions, sub_layers=True))  # build pd DataFrame to store results
            results.index.name = 'trial'
            # SETUP END #
            # run original model ***DONT RUN ORIG MODEL FOR SUBLAYERS - NOT NECESSARY
            # _, amp_mis, freq_mis = self.misfit(elastic=elastic, cadet_correct=cadet_correct,
            #                                   fill_troughs_pct=fill_troughs_pct,
            #                                   i_ang=i_ang, x_cor_range=x_cor_range, motion=motion,
            #                                   konno_ohmachi=konno_ohmachi, log_sample=log_sample)
            # if debug:
            #    self.misfit(elastic=elastic, cadet_correct=cadet_correct,
            #                fill_troughs_pct=fill_troughs_pct,
            #                i_ang=i_ang, x_cor_range=x_cor_range, plot_on=True, motion=motion,
            #                konno_ohmachi=konno_ohmachi, log_sample=log_sample)

            # print("Trial:{0}-Model:{1}-Misfit:({2},{4})-N_sub_layers:{3}".format(0, orig_sub.tolist() + [
            #    self.Mod['Qs'][0]],
            #                                                                     np.round(amp_mis, 3), n_layers,
            #                                                                     np.round(freq_mis, 3)))
            # store result in pandas data frame
            #results.loc[0] = orig_sub.tolist() + [self.Mod['Qs'][0]] + [amp_mis, freq_mis, n_layers]
            # loop over the model realisations picked at random and calculate misfit
            for i, model in enumerate(realisations):
                # print(model)
                if const_q is None:
                    self.modify_site_model(model, q_model=True)
                else:
                    self.modify_site_model(model)  # change the model in Valerio's SiteModel class
                # calculate misfit
                _, amp_mis, freq_mis, max_xcor = self.misfit(elastic=elastic, cadet_correct=cadet_correct,
                                                   fill_troughs_pct=fill_troughs_pct, i_ang=i_ang,
                                                   x_cor_range=x_cor_range, motion=motion, konno_ohmachi=konno_ohmachi,
                                                   log_sample=log_sample)
                if debug:
                    self.misfit(elastic=elastic, cadet_correct=cadet_correct,
                                fill_troughs_pct=fill_troughs_pct, i_ang=i_ang, x_cor_range=x_cor_range, plot_on=True,
                                motion=motion, konno_ohmachi=konno_ohmachi, log_sample=log_sample)
                # store result in data frame
                results.loc[i] = model.tolist() + [amp_mis, freq_mis, max_xcor, n_layers]
                print("Trial:{0}-Model:{1}-Misfit:({2},{4})-N_sub_layers:{3}".format(i, model, np.round(amp_mis, 3),
                                                                                     n_layers, np.round(freq_mis, 3)))

            if save:  # save the file as csv
                # results.to_csv(self.simulation_path+'.csv')
                # print('Need to add save clause: returning Data-Frames')
                results.to_csv(self.simulation_path + 'n_sub_' + str(n_layers) + '.csv')
            else:
                all_results.append(results)
        if not save:
            return results

    def reset_site_model(self, custom_model=None, q_model=False):
        """
        Reset the model to the original site model.
        :return: None - inplace method
        """
        self.Mod = {'Dn': [], 'Hl': [], 'Qp': [], 'Qs': [], 'Vp': [], 'Vs': []}
        self.__add_site_profile(custom_model=custom_model, q_model=q_model)
        self.vp_vs = np.array(self.Mod['Vp']) / np.array(self.Mod['Vs'])

    def read_post_sim_data(self, directory, run=0, its=5000, rtn=False):
        """
        
        :param directory: 
        :param run: 
        :param its: 
        :param rtn: 
        :return: 
        """
        directory = '/data/share/Japan/SiteInfo/S_B/' + directory
        self.sim_pars = parse_simulation_cfg(glob.glob(directory+'*{1}_run_{0}*.cfg'.format(run, its))[0])
        self.sim_results = [pd.read_csv(df, index_col=0) for df in
                            sorted(glob.glob(directory + '*{1}_run_{0}*.csv'.format(run, its)))]
        print('Found {0} tables.'.format(len(self.sim_results)))
        if rtn:
            return self.sim_results
