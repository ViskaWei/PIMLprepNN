from abc import ABC, abstractmethod
from base.interface.gateway.baseprocessIF import ProcessIF
from prepnn.center.crust.data.baseprepnn import PrepNN, StellarPrepNN
from prepnn.center.crust.process.prepnnprocess import StellarPrepNNProcess
from .prepnnloaderIF import StellarPrepNNLoaderIF
from .prepnnstorerIF import StellarPrepNNStorerIF
from .prepnnparamIF import StellarPrepNNParamIF

class PrepNNProcessIF(ProcessIF):
    @abstractmethod
    def interact_on_Object(self, NNP: PrepNN):
        super().interact_on_Object(NNP)

class StellarPrepNNProcessIF(PrepNNProcessIF):
    def __init__(self) -> None:
        super().__init__()
        self.Loader  = StellarPrepNNLoaderIF()   
        self.storer  = StellarPrepNNStorerIF()
        self.Process = StellarPrepNNProcess()
        self.Param   = StellarPrepNNParamIF()

    def interact_on_Object(self, NNP: StellarPrepNN):
        super().interact_on_Object(NNP)
