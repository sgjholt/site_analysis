import sys

sys.path.insert(0, '../SeismicSiteTool/')
import site_class as sc
import libs.SiteModel as sm


class Sim1D(sc.Site, sm.Site1D):
    def __init__(self, name, working_directory, vel_file_dir=None):
        super(Sim1D, self).__init__(name, working_directory, vel_file_dir)

    def add_site_profile(self):
        self.get_velocity_profile()
        pass
