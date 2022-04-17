from base.interface.gateway.baseparamIF import ParamIF
from prepnn.center.crust.data.constants import Constants


class StellarPrepNNParamIF(ParamIF):
    def set_param(self, PARAM):
        self.ntrain = self.get_arg("ntrain", PARAM, 10000)
        self.ntest  = self.get_arg("ntest",  PARAM, 100)
        self.seed   = self.get_arg("seed",   PARAM, None)

        self.INTERP_PATH = self.get_arg("INTERP_PATH", PARAM, Constants.INTERP_PATH)
        self.dim         = self.get_arg("dim", PARAM, 5)
        self.set_out(PARAM)
        self.set_param_dict()

    def set_param_dict(self):
        self.OBJECT = {
            "INTERP_PATH": self.INTERP_PATH,
            "dim"        : self.dim
        }
        self.OP = {
            "ntrain": self.ntrain,
            "ntest" : self.ntest,
            "seed"  : self.seed
        }

    def set_out(self, PARAM):
        self.name = self.get_arg("name", PARAM, "bosz_prepnn")
        self.dir = self.get_arg("dir", PARAM, Constants.STELLAR_PREPNN_DIR)
        out = {
            "name": self.name,
            "dir" : self.dir
        }
        self.OUT = out

        