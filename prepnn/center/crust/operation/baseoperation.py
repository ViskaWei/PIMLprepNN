from base.center.crust.baseoperation import BaseOperation, SamplingOperation

class LabelPrepOperation(BaseOperation):
    def __init__(self, ntrain, ntest, seed=None) -> None:
        self.ntrain = ntrain
        self.ntest = ntest
        self.seed = seed

    def perform(self, train_sampler, test_sampler=None):
        if test_sampler is None: test_sampler = train_sampler
        train_label = train_sampler(self.ntrain, seed=self.seed)
        test_label  = test_sampler(self.ntest, seed=self.seed)
        return train_label, test_label
        
class DataPrepOperation(BaseOperation):
    def perform(self, unit_coord, interpolator, noiser):
        data  = interpolator(unit_coord)
        sigma = noiser(data)
        return data, sigma

class LabelSamplerOperation(BaseOperation):
    def perform(self, coordx_dim, method):
        sampler = SamplingOperation(method).perform(coordx_dim)
        return sampler

