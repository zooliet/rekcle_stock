
import numpy as np
import pandas as pd
import logging

# Constant
UP = 1
DN = 0

class LabelGenerator:
    def __init__(self, logger=None):
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

    def process(self, df, target='Close'):
        df = df.copy()
        # y_idx
        df['y_idx'] = df[target].shift(-1)
        df.loc[df.last_valid_index(), 'y_idx'] = df[target].iloc[-1]

        # y_udn
        df['y_udn'] = np.where((df['y_idx']>df[target]), UP, DN)

        # y_rtn_day
        df['y_rtn_day'] = np.log(df['y_idx']/df[target])*100
        df['y_rtn_day'] = df['y_rtn_day'].shift(1).fillna(0.)

        self.logger.debug("LabelGenerator: \n{}\n".format(np.round(df.head().append(df.tail()),4)))
        return df
