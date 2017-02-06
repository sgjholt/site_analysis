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


def define_model_space(original, variation_pct, steps):
    """
    Defines the model space to be searched from given original values. The model space range is defined by a percentage
    set by the user. The model space covers the whole range +to- this percentage of the original value in a set number
    of steps - also defined by the user.

    :param original: original model - list or np.ndarray of length N
    :param variation_pct: single percentage value for extremes of search (e.g 50) - int/float
    :param steps: single value for number of values to generate as a factor of 5,10 is optimal - int/float
    :return: np.ndarray matrix containing the defined model space
    """
    model_space = np.zeros((len(original), steps+1))
    for i, row in enumerate(original):
        low = row - row*variation_pct/100
        high = row + row*variation_pct/100
        model_space[i] = np.linspace(high, low, steps+1)

    return np.concatenate((np.matrix.round(model_space, 0), np.matrix.round(
        np.logspace(np.log10(50), np.log10(2), steps+1, base=10), 2)[None, :]))


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
    cols = ['v'+str(num+1) for num in range(dimensions)]
    [cols.append(title) for title in ('qs', 'rms')]
    return cols

