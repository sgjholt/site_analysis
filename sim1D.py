
from site_class import Site


class Sim1D(Site):
    def __init__(self, name, working_directory, vel_file_dir=None):
        super(Sim1D, self).__init__(name, working_directory, vel_file_dir)
