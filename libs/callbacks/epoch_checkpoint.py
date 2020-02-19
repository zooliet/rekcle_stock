
import pdb
import rlcompleter
pdb.Pdb.complete=rlcompleter.Completer(locals()).complete

import numpy as np
# import pandas as pd
import logging

import os
# from keras.callbacks import BaseLogger
from keras.callbacks import Callback

class EpochCheckpoint(Callback):
    def __init__(self, epoch_at=0, output_path='./models/Test', every=5, logger=None):
        # call the parent constructor
        super(Callback, self).__init__()

        self.epoch_at = epoch_at
        self.output_path = output_path
        self.every = every

    def on_epoch_end(self, epoch, logs={}):
        # pdb.set_trace()

        # check to see if the model should be serialized to disk
        if (self.epoch_at + 1) % self.every == 0:
            p = "{}_{:03d}.hdf5".format(self.output_path, self.epoch_at+1)
            self.model.save(p, overwrite=True)

        # increment the internal epoch counter
        self.epoch_at += 1
