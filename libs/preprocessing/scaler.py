
# https://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas
# https://stackoverflow.com/questions/16349389/grouping-data-by-value-ranges

import numpy as np
import pandas as pd
import logging
import re

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

class Scaler:
    def __init__(self, op, excluding=['Close'], logger=None):
        self.op = op
        self.excluding = excluding

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

    def process(self, df, scaler=None):
        columns = df.columns.to_list()
        for target in self.excluding:
            columns.remove(target)

        df = eval(f"self.{self.op}(df, columns, scaler)")
        self.logger.debug(f"Scaler with {self.op}:\n{np.round(df,4)}\n")
        return df

    def std(self, df, columns, scaler):
        values = df[columns].values
        if scaler:
            values = scaler.transform(values)
        else:
            self.scaler = StandardScaler()
            values = self.scaler.fit_transform(values)

        df = df.copy()
        df.loc[:, columns] = values
        return df

    def maxabs(self, df, columns, scaler):
        values = df[columns].values
        if scaler:
            values = scaler.transform(values)
        else:
            self.scaler = MaxAbsScaler()
            values = self.scaler.fit_transform(values)

        df = df.copy()
        df.loc[:, columns] = values
        return df
