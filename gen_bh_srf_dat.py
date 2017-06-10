import numpy as np
import pandas as pd
from utils import interp_smooth


sd = '/data/share/Japan/SiteInfo/non_lin/'
db = pd.read_csv('/data/share/Japan/Kik_catalogue.csv', index_col=0)
srf_pga = db.query("instrument=='Surface'").pga.values
bh_pga = db.query("instrument=='Borehole'").pga.values
db = db.query("instrument=='Borehole'")
freqs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                  16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
cols = ['path', 'site', 'jmamag', 'Repi', 'Rhypo', 'eq_depth', 'Vs30', 'srf_pga', 'bh_pga', 'amp']

# pga_df = dict(zip(cols, [db[title].values for title in cols[:-3]]+[srf_pga, bh_pga, srf_pga/bh_pga]))
# pga_df = pd.DataFrame.from_dict(pga_df)
# pga_df.to_csv(sd+'PGA.csv')

FAS_dbs = []
for f in freqs:
    FAS_dbs.append(pd.DataFrame(columns=cols))

for j, item in enumerate(zip(db['path'], db['site'], db['jmamag'], db['Repi'], db['Rhypo'], db['eq_depth'],
                                 db['Vs30'])):
    FAS, _ = interp_smooth(item[0], sb=False, freqs=freqs)
    for i, _ in enumerate(freqs):
        FAS_dbs[i].loc[j] = [thing for thing in item]+[FAS[0][i], FAS[1][i], FAS[0][i]/FAS[1][i]]
    if (((j+1)*100)/len(db)) % 5 == 0:
        print('-------{0}% completion------'.format((((j+1)*100)/len(db))))

for i, df in enumerate(FAS_dbs):
    FAS_dbs[i].to_csv(sd+'FAS_{0}HZ.csv'.format(freqs[i]))



