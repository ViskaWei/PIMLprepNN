from abc import ABC, abstractmethod

class BasePrepNN(ABC):
    @abstractmethod
    def dim(self, dim):
        pass

class PrepNN(BasePrepNN):
    def __init__(self) -> None:
        self.train = {}
        self.test  = {}
        self.ntrain = 0
        self.ntest  = 0
        self.seed   = None

    def dim(self):
        return self.dim

class StellarPrepNN(PrepNN):
    def __init__(self, interpolator, noiser, label_rescaler, dim):
        PrepNN.__init__(self)
        
        self.interpolator   = interpolator
        self.noiser         = noiser
        self.label_rescaler = label_rescaler
        self.dim            = dim
        self.sampler        = {}
