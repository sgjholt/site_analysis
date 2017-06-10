import numpy as np
import pandas as pd
from utils import interp_smooth

if __name__ == "__main__":
    sd = '/data/share/Japan/SiteInfo/non_lin/'
    db = pd.read_csv('/data/share/Japan/Kik_catalogue.csv', index_col=0)
    # srf_pga = db.query("instrument=='Surface'").pga.values
    # bh_pga = db.query("instrument=='Borehole'").pga.values
    db = db.query("instrument=='Borehole'")
    freqs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                      16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    meta_cols = ['path', 'site', 'jmamag', 'Repi', 'Rhypo', 'eq_depth', 'Vs30']
    dat_cols = ['srf_FAS', 'bh_FAS', 'amp']

    # pga_df = dict(zip(cols, [db[title].values for title in cols[:-3]]+[srf_pga, bh_pga, srf_pga/bh_pga]))
    # pga_df = pd.DataFrame.from_dict(pga_df)
    # pga_df.to_csv(sd+'PGA.csv')

    FAS_dicts = []
    for f in freqs:
        FAS_dicts.append(dict(zip(dat_cols, [[], [], []])))

    for j, path in enumerate(db['path'].values[:10]):
        if (((j+1)*100)/len(db)) % 5 == 0:
            print('-------{0}% completion------'.format((((j+1)*100)/len(db))))
        skip = False
        try:
            FAS, _ = interp_smooth(path, sb=False, freqs=freqs)
        except (ValueError, IndexError):
            skip = True
        if skip:
            print('Error in filepath: '+path)
            for i, _ in enumerate(freqs):
                FAS_dicts[i]["srf_FAS"].append(999.0)
                FAS_dicts[i]["bh_FAS"].append(999.0)
                FAS_dicts[i]["amp"].append(999.0)

        else:
            for i, _ in enumerate(freqs):
                FAS_dicts[i]["srf_FAS"].append(FAS[0][i])
                FAS_dicts[i]["bh_FAS"].append(FAS[1][i])
                FAS_dicts[i]["amp"].append(FAS[0][i]/FAS[1][i])

    for i, f in enumerate(freqs):
        pd.DataFrame.from_dict({**dict(zip(meta_cols, [db[title].values[:10] for title in meta_cols])),
                                **FAS_dicts[i]}).to_csv(sd+'FAS_{0}HZ.csv'.format(f))
