import numpy as np
import pandas as pd
import random


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

    #bins = np.linspace(0, np.floor(max), (np.floor(max) / bin_width)+1)

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

def define_model_space(original, variation_pct, total_units):

    model_space = np.zeros((len(original), total_units))
    for i, row in enumerate(original):
        low = row - row/variation_pct
        high = row + row/variation_pct
        model_space[i] = np.concatenate(np.linspace(low, row, total_units/2), np.linspace(row, high, total_units/2)
