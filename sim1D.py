
import site_class as sc


class Sim1D(sc.Site):
    def __init__(self, name, working_directory, vel_file_dir=None):
        super(Sim1D, self).__init__(name, working_directory, vel_file_dir)
