
import numpy as np
# import pandas as pd
import logging

import json
import os
from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt

class TrainingMonitor(BaseLogger):
    def __init__(self, epoch_at=0, output_path='./output/test', logger=None):
        # store the output path for the figure, the path to the JSON
        # serialized file, and the starting epoch
        super(TrainingMonitor, self).__init__()
        self.epoch_at = epoch_at
        self.json_path = "{}.json".format(output_path)
        self.fig_path = "{}.png".format(output_path)

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)


    def on_train_begin(self, logs={}):
        # initialize the history dictionary
        self.H = {}

        if self.epoch_at == 0:
            return
        # if the JSON history path exists, load the training history
        if os.path.exists(self.json_path):
            self.H = json.loads(open(self.json_path).read())

            # check to see if a starting epoch was supplied
            if self.epoch_at > 0:
                # loop over the entries in the history log and
                # trim any entries that are past the starting
                # epoch
                for k in self.H.keys():
                    self.H[k] = self.H[k][:self.epoch_at]

    def on_epoch_end(self, epoch, logs={}):
        # loop over the logs and update the loss, accuracy, etc.
        # for the entire training process
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(float(v))
            self.H[k] = l

        # check to see if the training history should be serialized to file
        if self.json_path is not None:
            f = open(self.json_path, "w")
            f.write(json.dumps(self.H))
            f.close()

        # ensure at least two epochs have passed before plotting
        # (epoch starts at zero)
        if len(self.H["loss"]) > 1:
            # plot the training loss and accuracy
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure(figsize=(16,9))
            for key in self.H.keys():
                plt.plot(N, self.H[key], label=key)

            # plt.plot(N, self.H["loss"], label="train_loss")
            # if self.H.get('val_loss'):
            #     plt.plot(N, self.H["val_loss"], label="val_loss")
            # if self.H.get('acc'):
            #     plt.plot(N, self.H["acc"], label="train_acc")
            # if self.H.get('val_acc'):
            #     plt.plot(N, self.H["val_acc"], label="val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()

            # save the figure
            plt.savefig(self.fig_path)
            plt.close()
