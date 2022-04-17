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

    def finish(self, ext=".h5"):
        self.store_NN_preped(self.Object.train, self.Object.train_name, ext)
        self.store_NN_preped(self.Object.test,  self.Object.test_name,  ext)
        
    def store_NN_preped(self, data, name, ext):
        self.store.set_dir(self.OP_OUT["path"], store_name, ext)
        self.store.store_dict_args(data)