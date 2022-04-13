import numpy as np
from abc import ABC, abstractmethod
from base.center.crust.baseoperation import BaseOperation, BuildScalerOperation

from ..data.baseprepnn import PrepNN
from ..operation.baseoperation import LabelSamplerOperation, LabelPrepOperation, DataPrepOperation

class BasePrepNNOperation(BaseOperation):
    """ Base class for Process of preparing NN data. """
    @abstractmethod
    def perform_on_PrepNN(self, NNP: PrepNN): 
        pass

class LabelSamplerPrepNNOperation(LabelSamplerOperation, BasePrepNNOperation):
    def perform_on_PrepNN(self, NNP: PrepNN): 
        NNP.sampler = {}
        for method in ["halton", "uniform"]:
            NNP.sampler[method] = self.perform(NNP.dim, method)

class LabelPrepNNOperation(LabelPrepOperation, BasePrepNNOperation):
    def perform_on_PrepNN(self, NNP: PrepNN): 
        NNP.ntrain = self.ntrain
        NNP.ntest  = self.ntest
        NNP.seed   = self.seed
        
        NNP.train["label"], NNP.test["label"] = self.perform(NNP.sampler["uniform"], NNP.sampler["halton"])

class DataPrepNNOperation(DataPrepOperation, BasePrepNNOperation):
    def perform_on_PrepNN(self, NNP: PrepNN): 
        label       = np.vstack((NNP.train["label"], NNP.test["label"]))
        data, sigma = self.perform(label, NNP.interpolator, NNP.noiser)

        NNP.train["data"] , NNP.test["data"]  = data [:NNP.ntrain], data [NNP.ntrain:]
        NNP.train["sigma"], NNP.test["sigma"] = sigma[:NNP.ntrain], sigma[NNP.ntrain:]

#------------------------------------------------------------------------------
class NoiserPrepNNOperation(BasePrepNNOperation):
    def perform(self, Obs):
        noiser = lambda x: Obs.get_log_sigma(x, log=1)
        return noiser

    def perform_on_PrepNN(self, NNP: PrepNN):
        NNP.noiser = self.perform(NNP.Obs)

class LaberScalerPrepNNOperation(BuildScalerOperation, BasePrepNNOperation):
    def __init__(self, coordx_rng) -> None:
        super().__init__(0, coordx_rng)

    def perform_on_PrepNN(self, NNP: PrepNN): 
        self.get_scalers()
        NNP.coordx = self.tick
        NNP.coordx_dim = len(self.tick)
        NNP.label_scaler = self.scaler
        NNP.label_rescaler = self.rescaler