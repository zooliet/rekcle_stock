
import numpy as np
import pandas as pd
import logging
import re

class Classifier:
    def __init__(self, targets, boundaries, op, ref='close', logger=None):
        self.targets = targets
        self.op = op
        self.boundaries = boundaries
        self.ref = ref
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

        df = eval("self.with_{}(df, columns)".format(self.op))
        self.num_classes = len(self.boundaries) + 1
        self.logger.debug("Classifier on {}:\n{}\n".format(columns, np.round(df.head(),4)))
        return df

    def with_next_tick(self, df, columns):
        ref = df[self.ref].values.reshape(-1,1)
        y = (df[columns].values - ref) * 100 # ratio to percent
        y = np.digitize(y, self.boundaries)
        df['y'] = y[:,0:1]
        df.drop(columns, inplace=True, axis=1)
        return df

    def with_last_tick(self, df, columns):
        ref = df[self.ref].values.reshape(-1,1)
        y = (df[columns].values - ref) * 100 # ratio to percent
        y = np.digitize(y, self.boundaries)
        df['y'] = y[:,-1:]
        df.drop(columns, inplace=True, axis=1)
        return df

    def with_multi_ticks(self, df, columns):
        ref = df[self.ref].values.reshape(-1,1)
        y = (df[columns].values - ref) * 100 # ratio to percent
        y = np.digitize(y, self.boundaries)
        df[columns] = y
        return df

    def with_trending(self, df, columns):
        self.boundaries = [-0.5, 0.5]
        ref = df[self.ref].values.reshape(-1,1)
        y = (df[columns].values - ref) * 100 # ratio to percent
        y = np.digitize(y, self.boundaries)
        offset = len(self.boundaries) // 2  # 중간값(상승, 하락이 모두 아닌 값)을 찾기 위한 조치
        ups = (y > offset).all(axis=1).astype(np.int)
        downs = (y < offset).all(axis=1).astype(np.int) * (-1)
        y = ups + downs + 1  # 0 for downs, 2 for up, 1 for else
        df['y'] = y
        df.drop(columns, inplace=True, axis=1)
        return df

    def with_last_threes(self, df, columns):
        ref = df[self.ref].values.reshape(-1,1)
        y = df[columns].values[:,-3:].mean(axis=1).reshape(-1,1)
        y = (y - ref) * 100
        y = np.digitize(y, self.boundaries)
        df['y'] = y
        df.drop(columns, inplace=True, axis=1)
        return df
