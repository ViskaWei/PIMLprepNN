from abc import ABC, abstractmethod

from prepnn.center.crust.data.baseprepnn import PrepNN, StellarPrepNN
from ..operation.prepnnoperation import BasePrepNNOperation,\
    LabelSamplerPrepNNOperation, NoiserPrepNNOperation, LabelPrepNNOperation,\
    LaberScalerPrepNNOperation, DataPrepNNOperation

from base.center.crust.baseprocess import BaseProcess




class PrepNNProcess(BaseProcess):
    def __init__(self) -> None:
        self.operation_list: list[BasePrepNNOperation] = None
    def set_process(self):
        pass
    def start(self, NNP: PrepNN):
        for operation in self.operation_list:
            operation.perform_on_PrepNN(NNP)
                

    

class StellarPrepNNProcess(PrepNNProcess):

    def set_process(self, PARAMS, MODEL, DATA):
        self.operation_list = [
            LabelSamplerPrepNNOperation(),
            LabelPrepNNOperation(PARAMS["ntrain"],PARAMS["ntest"], PARAMS["seed"]),
            DataPrepNNOperation(),
        ]

    def start(self, NNP: StellarPrepNN):
        super().start(NNP)
