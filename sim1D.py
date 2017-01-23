import sys

sys.path.insert(0, '../SeismicSiteTool/')
import site_class as sc
import libs.SiteModel as sm


class Sim1D(sc.Site, sm.Site1D):

    def __init__(self, name, working_directory, vel_file_dir=None):
        sc.Site.__init__(self, name, working_directory, vel_file_dir)
        sm.Site1D.__init__(self)
        self.__add_site_profile()

    def __add_site_profile(self):
        vels = self.get_velocity_profile()
        for i, hl in enumerate(vels['thickness']):
            self.AddLayer([hl, vels['vp'][i], vels['vs'][i], vels['rho'][i], 1, 1])
