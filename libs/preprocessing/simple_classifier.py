
import numpy as np
import pandas as pd
import logging
import re

class SimpleClassifier:
    def __init__(self, targets, ref='Close', logger=None):
        self.targets = targets
        self.ref = ref
        self.num_classes = 2  # hard coded
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

    def process(self, df):
        # columns = list(filter(lambda name: self.target in name, df.columns))
        column_names = df.columns.to_list()
        columns = []
        for target in self.targets:
            # columns += list(filter(lambda name: target in name, column_names))
            columns += list(filter(lambda name: re.match(target, name), column_names))

        ref = df[self.ref].values.reshape(-1,1)
        y = np.where((df[columns].values > ref), 1, 0)
        df['y'] = y[:, -1:] # or df['y'] = y[:,0:1]

        self.logger.debug(f"Classifier:\n{np.round(df,4)}\n")
        return df
