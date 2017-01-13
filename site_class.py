import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Site:
    """Site helper class for visulaising and comparing site conditions.
     USAGE: var = site('SITE_CODE_HERE', pard='/user/home/one_higher_than_S_B_folder')
     Attributes: parent_directory - defaults to seismo-0's if none given
                 site - name of the site given during __init__ phase (required) - type str
     Methods: sb_ratio() - grabs the sb_ratio """

    def __init__(self, name, pard=None):
        if pard is not None:
            self.parent_directory = pard
        else:
            self.parent_directory = '/data/share/Japan/SiteInfo/'
        self.site = name

    def sb_ratio(self):
        return pd.read_csv(self.parent_directory + 'S_B/' + self.site + '.csv', index_col=0)

    def plot_sb(self, stdv=None, pctile=None, show=True):
        """The plot_sb method allows you to plot the S/B ratio calculated for the Site class object.
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
        plt.legend()
        if show:
            plt.show()

    def qc(self, plot_on=False):
        code = can_qc(self.site)
        if code is None:
            if plot_on:
                self.plot_sb()


def can_qc(site_name):
    with open('/home/james/Dropbox/site_ave/stcodeconv.dat') as site_codes:
        ben_site_dict = {sc.split()[0]: sc.split()[1] for sc in site_codes}

    possible = True
    try:
        list(ben_site_dict.keys()).index(site_name)
    except ValueError:
        possible = False

    if possible:
        return ben_site_dict['site_name']
    else:
        print('QC not available')
        return None


def parse_ben_sb(site, code_dict):
    bens = np.loadtxt(code_dict[site]+'res_ave.out')

