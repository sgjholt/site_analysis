# site_class - Base Site class object - handles site metadata - produces plots and other useful helper functions.
# VERSION: 0.9
# AUTHOR(S): JAMES HOLT - UNIVERSITY OF LIVERPOOL
#
# EMAIL: j.holt@liverpool.ac.uk
#
#
# ---------------------------------modules--------------------------------------#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from parsers import parse_ben_sb, read_kik_vel_file, parse_metadata, parse_litho
from utils import calc_density_profile, dest_freq, downgoing_transform_func


# ---------------------------------classes--------------------------------------#


class Site:
    """
    Site helper class for visualising and comparing site conditions.

    USAGE: var = site('SITE_CODE_HERE', pard='/user/home/one_higher_than_S_B_folder')

    Attributes: working_directory - directory where files are stored

                self.site - name of the site given during __init__ phase (required) - type str

                self.vel_file_dir - directory where kik-net vel files are located - defaults to my server location if none
                given - type str

                self.metadata_path - path to Kik-Net metadata file - obtained from the Kik-Net website
                http://www.kyoshin.bosai.go.jp/ - type str

                self.metadata - dict object containing metadata for the given site

    Methods: sb_ratio() - grabs the sb_ratio
     """
    working_directory = ''
    site = ''
    vel_file_dir = '/data/share/Japan/SiteInfo/physicaldata/'
    metadata_path = '/data/share/Japan/SiteInfo/sitepub_kik_en.csv'
    metadata = {}
    has_vel_profile = True
    vp_vs = None

    def __init__(self, name, working_directory, vel_file_dir=None, metadata_path=None):

        self.working_directory = working_directory
        self.site = name
        if vel_file_dir is not None:
            self.vel_file_dir = vel_file_dir
        if metadata_path is not None:
            self.metadata_path = metadata_path
        self.metadata = parse_metadata(self.metadata_path, name)
        if self.get_velocity_profile() is None:
            self.has_vel_profile = False
        else:
            if np.any(self.get_velocity_profile()['vs'] == 0):
                self.has_vel_profile = False
            else:
                pass
        if self.has_vel_profile:
            prof = self.get_velocity_profile()
            self.vp_vs = prof['vp']/prof['vs']

    def get_velocity_profile(self, litho=False):
        """
        get_velocity_profile() grabs the velocity profile provided by Kik-Net (if one exists) and returns numpy.ndarray
        object with N*M dimensions. Refer to read_kik_vel_file in parsers for detailed info.

        ***NOTE***: some minor processing occurs in parser function before returning - see above for details.

        USAGE: vel_profile = Site.get_velocity_profile()
            - vel_profile[:,0] = index
            - vel_profile[:,1] = Thickness [m]
            - vel_profile[:,2] = Depth [m]
            - vel_profile[:,3] = Vp [m/s]
            - vel_profile[:,4] = Vs [m/s]
        """
        vel_profile = None
        titles = ['thickness', 'depth', 'vp', 'vs']

        if not litho:  # load standard velocity model
            try:
                vel_profile = read_kik_vel_file(self.vel_file_dir + 'velocity_models/' + self.site + '.dat')
            except FileNotFoundError:
                print('No velocity profile for ' + self.site)
                return vel_profile

            vel = {titles[i]: vel_profile[:, i + 1] for i in range(len(titles))}
            vel.update({'rho': calc_density_profile(vel_profile[:, 3] / 1000) * 1000, 'rho_sig': 130})

        if litho:  # load custom lithology based model (defined by me)
            try:
                vel_profile = parse_litho(self.vel_file_dir + 'litho_vel_models/' + self.site + '.litho.csv')
            except FileNotFoundError:
                print('No velocity profile for ' + self.site)
                return vel_profile

            vel = {titles[i]: vel_profile[1][:, i] for i in range(len(titles))}
            vel.update({'rho': calc_density_profile(vel_profile[1][:, 3] / 1000) * 1000, 'rho_sig': 130})
            vel.update({'type': list(vel_profile[0])})

        return vel

    def sb_ratio(self, mod_factor=None, cadet_correct=False, prof=None, litho=False):
        sb = pd.read_csv(self.working_directory + self.site + '.csv', index_col=0)

        if prof is not None:
            profile = prof
        else:
            profile = self.get_velocity_profile(litho=litho)

        if cadet_correct:  # apply correction detailed in Cadet 2011 - see function
            cor = downgoing_transform_func(np.array(sb.columns.values, dtype=float), dest_freq(
                self.metadata['depth'], profile))
            sb *= cor
            sb.loc['count'] /= cor

        if mod_factor is not None:
            sb *= mod_factor
            sb.loc['count'] /= mod_factor

        return sb

    def plot_sb(self, stdv=None, pctile=None, show=True, cadet_correct=False, mod_factor=None):
        """
        The plot_sb method allows you to plot the S/B ratio calculated for the Site class object.
        USAGE: Site.plot_sb(stdv=tuple of ints, pctile=25, 50 or 75% as tuple of str , show=True)
        """
        table = self.sb_ratio(mod_factor=mod_factor, cadet_correct=cadet_correct)
        plt.loglog(table.columns.values, np.exp(table.loc['mean']), 'k', label='mean')
        plt.hlines(1, float(table.columns.values[0]), float(table.columns.values[-1]), colors='k', linestyles='dashed')
        plt.title(self.site+': S/B Ratio - {0} records'.format(int(table.loc['count'][0])))
        opts = ('--k', '--r', '--b')
        if stdv is not None:
            counter = 0
            for num in stdv:
                if counter == 0:
                    plt.loglog(table.columns.values, np.exp(
                        table.loc['mean'] + table.loc['std'] * num), '--k', label='std')
                    plt.loglog(table.columns.values, np.exp(
                        table.loc['mean'] - table.loc['std'] * num), '--k')
                    # plt.loglog(table.columns.values, np.exp(
                    #    table.loc['mean'] + table.loc['mean'] * table.loc['std']*num), '--k', label='std')
                    # plt.loglog(table.columns.values, np.exp(
                    #    table.loc['mean'] - table.loc['mean'] * table.loc['std']*num), '--k')

                else:
                    plt.loglog(table.columns.values, np.exp(
                        table.loc['mean'] + table.loc['std'] * num), opts[counter], label='std')
                    plt.loglog(table.columns.values, np.exp(
                        table.loc['mean'] - table.loc['std'] * num), opts[counter])
                    # plt.loglog(table.columns.values, np.exp(
                    #    table.loc['mean'] + table.loc['mean'] * table.loc['std'] * num), '--k')
                    # plt.loglog(table.columns.values, np.exp(
                    #    table.loc['mean'] - table.loc['mean'] * table.loc['std'] * num), '--k')
                counter += 1
        if pctile is not None:
            counter = 0
            for pc in pctile:
                plt.loglog(table.columns.values, np.exp(table.loc['mean'] + table.loc[pc]), opts[counter], label=pc)
                plt.loglog(table.columns.values, np.exp(table.loc['mean'] - table.loc[pc]), opts[counter])
                # plt.loglog(table.columns.values, np.exp(table.loc['mean'] + table.loc['mean'] * table.loc[pc]), '--k')
                # plt.loglog(table.columns.values, np.exp(table.loc['mean'] - table.loc['mean'] * table.loc[pc]), '--k')
                counter += 1
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('S/B Ratio')
        plt.legend(loc=3)
        if show:
            plt.show()
        else:
            pass

    def qc(self, plot_on=True, cadet_correct=False, show=False):
        """
        The method qc allows visual (and soon statistical) comparison of mine and Ben's empirical S/B ratios
         calculated for the Kik-Net network.

        USAGE: Site.qc(plot_on=True/False, show=True/False)

        The default arguments allow plotting multiple graphs as they pass implicitly to the matplotlib
        object which is created upon the first instance of calling plt.loglog(). Refer to function rand_qc() in utils.py
        for idea of usage.
        """
        code = can_qc('/home/sgjholt/site_ave/', self.site)
        if code is not None:
            if plot_on:
                ben = parse_ben_sb('/home/sgjholt/site_ave/', code)
                plt.loglog(ben[:, 0], ben[:, 1], 'r', label='Ben')
                self.plot_sb(stdv=(1,), cadet_correct=cadet_correct, show=show)

            else:
                pass
        else:
            pass

    def add_metadata(self, metadata_dict):
        self.metadata.update(metadata_dict)


# ---------------------------------functions--------------------------------------#


def can_qc(parent_directory, site_name):
    """
    can_qc() is a helper function for the Site class object. It will determine if it is possible to quality control
    my results against Ben's. If it is possible it returns a numeric site code from Ben's dictionary (stcodeconv.dat)
    to point to the corresponding data file for a given site. If the site doesn't exist in Ben's dictionary it will
    return a None type object and print to screen for troubleshooting purposes.

    USAGE: can_qc('directory_to_file_containing_dictionary_and_data', 'site_name')
    """
    with open(parent_directory+'stcodeconv.dat') as site_codes:
        ben_site_dict = {sc.split()[0]: sc.split()[1] for sc in site_codes}

    possible = True
    try:
        list(ben_site_dict.keys()).index(site_name)
    except ValueError:
        possible = False

    if possible:
        return ben_site_dict[site_name]
    else:
        print('QC not available for ' + site_name)
        return None
