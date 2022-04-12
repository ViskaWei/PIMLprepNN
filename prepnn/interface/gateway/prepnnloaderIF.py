import os
from base.interface.gateway.baseloaderIF import DataLoaderIF, ObjectLoaderIF
from prepnn.center.crust.data.baseprepnn import PrepNN, StellarPrepNN

class PrepNNLoaderIF(ObjectLoaderIF):
    def load(self) -> PrepNN:
        pass

class StellarPrepNNLoaderIF(PrepNNLoaderIF):
    def set_param(self, PARAM):
        self.INTERP_PATH = PARAM["INTERP_PATH"]
        self.dim         = PARAM["dim"]

    def load(self) -> StellarPrepNN:
        interp = self.load_interp()
        noiser = self.load_noiser()
        return StellarPrepNN(interp, noiser, self.dim)

    def load_interp(self):
        interp = DataLoaderIF(self.INTERP_PATH).load()
        return interp

    def load_noiser(self):
        noiser = lambda x: 1 / ((x / 10) ** 0.5)
        # Obs = DataLoaderIF(self.OBS_PATH).load()
        # noiser = lambda x: Obs.get_log_sigma(x, log=1)
        return noiser


