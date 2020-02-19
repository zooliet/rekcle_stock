
import numpy as np
import logging
import re

class ColumnRemover:
    def __init__(self, targets, singleton=True, logger=None):
        self.targets = targets
        self.singleton = singleton
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

    def process(self, df):
        column_names = df.columns.to_list()
        columns = []
        for target in self.targets:
             # columns += list(filter(lambda name: target in name, column_names))
             columns += list(filter(lambda name: re.match(target, name), column_names))

        df.drop(columns, inplace=True, axis=1)
        self.logger.debug("ColumnRemover at {}:\n{}\n".format(self.targets, np.round(df,4)))
        return df
