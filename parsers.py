# Parsers used for the site_

import numpy as np
import pandas as pd


def parse_ben_sb(parent_directory, code):
    """
    parse_ben_sb() is a helper function for the Site class object. It uses numpy to load his data into the memory
    as an numpy.ndarry() object.

    USAGE: parse_ben_sb('directory_to_file_containing_dictionary_and_data', 'site_name')
    """
    s_b = np.loadtxt(parent_directory+code+'res_ave.out')
    return s_b


def read_kik_vel_file(fname):
    """Reads in a site velocity file in the SITE.dat format, returns
    numpy array containing the data without the headers. Some files are
    missing values in the first row and the bottom line always has '  -----,'
    for thickness depth. The function replaces blank lines with the velocity
    in the cells below and the thickness with 1000 m and depth with final
    depth measurement + 1000 m. Returned is a square N*M matrix where M is
    5 and N is arbitrary. Returned file has no headers - Col0 = Layer Num
    Col1 = Thickness (m), Col2 = Depth (m), Col3 = Vp (m/s) and Col4 = Vs (m/s)
    """
    with open(fname) as f:
        strings = [s.split(',') for s in [b.strip('\n') for b in f.readlines()]]

    if strings[-1][1] == '   -----':
        strings[-1][1] = '1000'
        strings[-1][2] = str(float(strings[-2][2]) + 1000)

        # sometimes true that you're missing values in top line (found 1 case so far)
    if strings[2][4] == '    0.00' or strings[2][4] == '        ':
        strings[3][1] = strings[3][2]  # make sure thickness is same as depth
        del strings[2]  # remove the whole line
    # always true, the last thickness/depth will always look like this

    # make sure lines below also contain relevent data, if not remove the line
    indices = [i for i, s in enumerate(strings) if '        ' in s]
    if len(indices) == 1 and indices[0] == (len(strings) - 1):
        del strings[indices[0]]

    # pre allocate numpy array and assign values to it
    dat = np.zeros(((len(strings) - 2), 5))
    for n in range(2, len(strings)):
        dat[n - 2, :] = [float(s) for s in strings[n]]

    return dat


def parse_ben_event_select(_file):
    strings = []
    with open(_file, 'rt') as f:
        for i, line in enumerate(f):
            strings.append(line.split()[0:2])
            strings[i] = strings[i][0][3:] + strings[i][1]
    return [int(string) for string in strings]


def parse_metadata(path, site):
    """
    parse_metadata() loads the Kik-Net site database and finds the information relevent to the site passed as an
    argument.

    :param path: str object containing path to the site database file e.g. '/home/user/sitepub_kik_en.csv'

    :param site: str object containing the site name e.g. 'IWTH06'

    :return: dict object containing site metadata
    """
    headers = ['name', 'lat', 'lon', 'altitude', 'depth', 'prefecture', 'otherlat', 'otherlon', 'instrument']
    site_info = pd.read_csv(path, index_col=0, header=None).loc[site].values[0:9]
    return {header: site_info[i] for i, header in enumerate(headers)}


def parse_litho(path):
    """
    parse_litho returns site.litho.csv a tuple containing separate numpy.ndarry info.
    litho_names are the names of each layer - str
    litho_vals are the thickness (m), depth (m), vp (m/s) and vs (m/s), column-wise, respectively
    Remember depth is representative of the bottom of the given layer

    :param path: string value containing full path to file to be parsed

    :return: tuple of np.ndarrays where tuple[0] is litho_names and tuple[1] is litho_vals

    """
    contents = []
    with open(path, 'rt') as f:
        for line in f:
            contents.append(line.strip('\r').strip('\n').split(','))
    litho_names = np.array(contents)[1:][:, 0]  # recast as np.ndarray for convenience
    litho_vals = np.array(contents)[1:].T[1:].T.astype(float)
    thickness = np.array(
        [litho_vals[:, 0][0]] + [litho_vals[:, 0][i] - litho_vals[:, 0][i - 1] for i in range(1, len(litho_vals))])
    litho_vals = np.concatenate((thickness.reshape(thickness.size, 1), litho_vals), axis=1)

    return litho_names, litho_vals
