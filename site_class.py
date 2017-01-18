

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from parsers import parse_ben_sb, read_kik_vel_file


class Site:
    """
    Site helper class for visualising and comparing site conditions.

    USAGE: var = site('SITE_CODE_HERE', pard='/user/home/one_higher_than_S_B_folder')

    Attributes: working_directory - directory where files are stored
                site - name of the site given during __init__ phase (required) - type str
                vel_file_dir - directory where kik-net vel files are located - defaults to my server location if none
                given
    # blah
    Methods: sb_ratio() - grabs the sb_ratio
     """
    working_directory = ''
    site = ''
    vel_file_dir = '/data/share/Japan/SiteInfo/physicalData/'

    def __init__(self, name, working_directory, vel_file_dir=None):

        self.working_directory = working_directory
        self.site = name
        if vel_file_dir is not None:
            self.vel_file_dir = vel_file_dir

    def get_velocity_profile(self):
        """
        get_velocity_profile() grabs the velocity profile provided by Kik-Net (if one exists) and returns numpy.ndarray
        object with N*M dimensions. Refer to read_kik_vel_file in parsers for detailed info.

        ***NOTE***: some minor processing occurs in parser function before returning - see above for details.

        USAGE: vel_profile = Site.get_velocity_profile()
            - vel_profile[:,0] = Thickness [m]
            - vel_profile[:,1] = Depth [m]
            - vel_profile[:,2] = Vp [m/s]
            - vel_profile[:,3] = Vs [m/s]
        """
        vels = None
        try:
            vels = read_kik_vel_file(self.vel_file_dir+self.site+'.dat')
        except FileNotFoundError:
            print('No velocity profile for '+self.site)
        return vels

    def sb_ratio(self):
        return pd.read_csv(self.working_directory + self.site + '.csv', index_col=0)

    def plot_sb(self, stdv=None, pctile=None, show=True):
        """
        The plot_sb method allows you to plot the S/B ratio calculated for the Site class object.
        USAGE: Site.plot_sb(stdv=tuple of ints, pctile=25, 50 or 75% as tuple of str , show=True)
        """
        table = self.sb_ratio()
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

    def qc(self, plot_on=True, show=False):
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
                self.plot_sb(stdv=(1,), show=show)
            else:
                pass
        else:
            pass


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




