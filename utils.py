import numpy as np
import pandas as pd
import random
import contextlib
import os


def rand_sites(sample_size):
    sites = sorted(pd.read_csv('/data/share/Japan/Kik_catalogue.csv', index_col=0).site.unique())
    return [sites[num] for num in random.sample(range(len(sites)-1), sample_size)]


# def test_loc_change():
    # uniques = {}
    # sites = sorted(list(set([x for x in ALL.site.values])))
    # for site in sites:
    #    subset = ALL.query('instrument == "Borehole" & site == \"{0}\"'.format(site))
    #    uniques.update({site: subset.site_lon.unique()})
    # return uniques


def calc_density_profile(Vp):
    """
    calc_density_profile() is a helper function for the Site class object. It will determine a theoretical density
    profile based on Dave Boore's empirical model (2016).

    See http://daveboore.com/pubs_online/generic_velocity_density_models_bssa_2016.pdf for details.
    Takes a numpy.ndarry object as input.

    The density profile is calculated using the Vp profile - the standard deviation is about 0.13 g/cm^3.

    USAGE: calc_density_profile(array([Vp]))

    :return : np.ndarray vector containing theoretical density values
    """
    return (1.6612 * Vp) - (0.4721 * Vp ** 2) + (0.0671 * Vp ** 3) - (0.0043 * Vp ** 4) + (0.000106 * Vp ** 5)


def binning(array, freqs, bin_width, even_spaced=True):
    """

    :param array: values to be binned (in the sense the mean value will be given) - numpy.ndarray
    :param freqs: frequencies used to bin data
    :param bin_width: the width of the bin in unit spacing between frequency values
    :param even_spaced: automatically determines the best next best width to use for even bins if the one provided is
        not adequate.
    :return: tuple containing the binned values [0] and the bins [1]
    """

    if even_spaced:
        while np.floor(freqs[-1]) % bin_width not in {0, 5}:
            bin_width += 0.5
            print(bin_width)
        print('next optimal bin_width is {0}'.format(bin_width))

    bin_freqs = np.linspace(freqs[0], freqs[-1], len(freqs)/bin_width+1)

    # bins = np.linspace(0, np.floor(max), (np.floor(max) / bin_width)+1)

    binned = np.zeros(int(len(bin_freqs)))

    for i, freq in enumerate(bin_freqs):
        if freq == bin_freqs[0]:
            binned[i] = np.mean(array[freqs < freq + bin_width / 2])
        elif freq == bin_freqs[-1]:
            binned[i] = np.mean(array[freqs >= freq - bin_width / 2])
        else:
            binned[i] = np.mean(array[(freqs >= freq - bin_width/2) & (freqs < freq + bin_width/2)])
    return binned, bin_freqs


def pick_model(model_space, perm):
    current_perm = []
    for j, row in enumerate(model_space):
        current_perm.append(row[perm[j]])
    return np.array(current_perm)


def uniform_model_space(original, variation_pct, steps, vs_only=False, const_q=None):
    """
    Defines the model space to be searched from given original values. The model space range is defined by a percentage
    set by the user. The model space covers the whole range +to- this percentage of the original value in a set number
    of steps - also defined by the user.

    :param original: original model - list or np.ndarray of length N
    :param variation_pct: single percentage value for extremes of search (e.g 50) - int/float
    :param steps: single value for number of values to generate as a factor of 5,10 is optimal - int/float
    :param const_q: constant value for Q if not None
    :return: np.ndarray matrix containing the defined model space
    """
    model_space = np.zeros((len(original), steps+1))
    for i, row in enumerate(original):
        low = row - row*variation_pct/100
        high = row + row*variation_pct/100
        model_space[i] = np.linspace(high, low, steps+1)

    if vs_only:
        return np.matrix.round(model_space, 0)

    if const_q is not None:
        return np.concatenate((np.matrix.round(model_space, 0), np.matrix.round(
            np.zeros(steps+1)+10, const_q)[None, :]))
    else:
        return np.concatenate((np.matrix.round(model_space, 0), np.matrix.round(
            np.logspace(np.log10(50), np.log10(2), steps+1, base=10), 2)[None, :]))


def correlated_sub_model_space(original, variation_pct, cor_pct, steps, const_q=None):
    """
    UNFINISHED
    :param original:
    :param variation_pct:
    :param cor_pct:
    :param steps:
    :param const_q:
    :return:
    """
    ufms = uniform_model_space(original, variation_pct, steps, vs_only=True)
    print('ufms={0}'.format(ufms))
    csms = np.zeros((len(ufms)*2, steps+1))
    csms[0] = ufms[0]  # allocate the first layer
    print('cfms[0]={0}, ufms[0]={1}'.format(csms[0], ufms[0]))

    for i in range(1, len(csms)):

        if i % 2 == 0:
            csms[i] = ufms[int(i/2)]
        else:
            csms[i] = csms[i-1]*(1+cor_pct/100)
        # if i >= 3 and i % 2 > 0 and (ufms[i-2]-ufms[i-3])/abs((ufms[i-2]-ufms[i-3]))[0] == -1:
        #    csms[i] = csms[i - 1] * (1 - cor_pct / 100)

    return csms


def silent_remove(filename):
    """
    The silent_remove() function uses the os.remove method to remove a
    given file. The contextlib module suppresses the error given if the file
    does not exist, allowing the current run to continue to execute.

    USAGE: silent_remove(path+filename)
    parent_dir = /home/directory_of_interest/
    fname = "file.txt"
    E.G. - silent_remove(parent_dir + fname)
    """
    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)


def df_cols(dimensions):
    cols = ['v'+str(num+1) for num in range(dimensions-1)]
    [cols.append(title) for title in ('qs', 'amp_mis', 'freq_mis', 'total_mis')]
    return cols


def dest_freq(bh_depth, prof):
    """
    Calculates the destructive frequency expected at a given site.
    See Cadet et al., 2011 : http://link.springer.com/article/10.1007/s10518-011-9283-1
    :param bh_depth: depth of the borehole instrument (m) - type int/float
    :param prof: site profile - dict - see site_class.py method - self.get_velocity_profile()
    :return: fundamental frequency (Hz)
    """
    if prof['depth'][-1] == bh_depth:  # if the depth of the final layer is the borehole depth (unlikely)
        mean_vel = prof['vs'][:-1].mean()  # mean vel above bh depth is mean of every vs above that point
    else:  # the last layer depth is > bh depth - take mean of all vs - bh exists in that last layer
        mean_vel = prof['vs'].mean()

    return mean_vel/(4*bh_depth)


def downgoing_transform_func(f, fd, A=1.8, B=0.8, sig=0.15):
    """
    Function to remove the effect of the artificial increase in relative amplitude in Surface/Borehole ratios.
    Due to destructive interference effects at depth.
    See Cadet et al., 2011 : http://link.springer.com/article/10.1007/s10518-011-9283-1

    :param f: frequencies to build calculate transfer function
    :param fd: expected destructive frequency for your site
    :param A: constant (set 1.8 as recommended by Cadet et al., 2011)
    :param B: constant (set 0.8 as recommended by Cadet et al., 2011)
    :param sig: constant (set 0.15 as recommended by Cadet et al., 2011)
    :return: transfer function
    """

    c1 = 1 + ((B * np.arctan(f / fd)) / (np.pi / 2))
    c2 = 1 + (A - 1) * np.exp(-(f / fd - 1) ** 2 / (2 * sig) ** 2)
    return 1/(c1*c2)


def exp_cdf(x, lam):
    """
    CDF of exponential distribution for given parameters x and lam(bda)

    :param x: Variable(s) - int/float/np.ndarray
    :param lam: Defines shape of CDF - int/float
    :return: Normalised CDF value for given x and lambda - float/np.ndarray
    """
    return 1-np.exp(-lam*x)

# import scipy.find_peaks_cwt as fp
# sb = self.sb_ratio()
# sb_mean = sb.loc['mean']
#

# plt.plot(np.array(fp(sb_mean, np.arange(1,11)))*0.01, np.exp(sb_mean[fp(sb_mean, np.arange(1,11))]), 'r*')
