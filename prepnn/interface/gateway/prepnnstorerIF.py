import os
from base.interface.gateway.basestorerIF import DictStorerIF, ObjectStorerIF
from prepnn.center.crust.data.baseprepnn import PrepNN, StellarPrepNN

class StellarPrepNNStorerIF(ObjectStorerIF):
    def set_param(self, PARAM):
        name = PARAM["name"]
        dir  = PARAM["dir"]
        self.TRAIN_PATH = os.path.join(dir, name + "_train.h5")
        self.TEST_PATH  = os.path.join(dir, name + "_test.h5")

    def store_train(self, NNP: StellarPrepNN):
        self.storer = DictStorerIF(self.TRAIN_PATH)
        self.storer.store_dict_args(NNP.train)

    def store_test(self, NNP: StellarPrepNN):
        self.storer = DictStorerIF(self.TEST_PATH)
        self.storer.store_dict_args(NNP.test)

    def store(self, NNP: StellarPrepNN):
        self.store_train(NNP)
        self.store_test(NNP)

