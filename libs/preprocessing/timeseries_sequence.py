
import numpy as np
import pandas as pd
import logging

class TimeseriesSequence:
    def __init__(self, step_in, step_out, offset=0, target='Close', logger=None):
        self.step_in = step_in
        self.step_out = step_out
        self.offset = offset
        self.target = target
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

    def sequence_generate(self, df):
        X, y = [], []
        targets = df[self.target].values
        sequences = df.values

        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + self.step_in
            end_outx = end_ix + self.step_out + self.offset
            # check if we are beyond the sequences
            if end_outx > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x = sequences[i:end_ix]
            # seq_y = sequences[end_ix:end_outx]
            seq_y = targets[end_ix+self.offset:end_outx]
            X.append(seq_x.reshape(-1,))
            y.append(seq_y.reshape(-1,))

        column_names = df.columns.to_list()

        x_names = []
        for i in range(self.step_in-1, 0, -1):
            for name in column_names:
                x_names.append("{}-{}".format(name, i))
        x_names = x_names + column_names

        y_names = []
        for i in range(1, self.step_out+1):
            y_names.append("{}+{}".format(self.target, i))

        df_X = pd.DataFrame(X, columns=x_names)
        df_y = pd.DataFrame(y, columns=y_names)
        df = df_X.join(df_y)

        self.logger.debug(f"TimeseriesSequences:\n{np.round(df,4)}\n")
        return df

    def target_sequence_generate(self, df):
        X, y = [], []
        targets = df[self.target].values
        sequences = df.values

        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + self.step_in
            end_outx = end_ix #+ self.step_out + self.offset
            # check if we are beyond the sequences
            if end_outx > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x = sequences[i:end_ix]
            X.append(seq_x.reshape(-1,))

        column_names = df.columns.to_list()

        x_names = []
        for i in range(self.step_in-1, 0, -1):
            for name in column_names:
                x_names.append("{}-{}".format(name, i))
        x_names = x_names + column_names

        df = pd.DataFrame(X, columns=x_names)

        self.logger.debug(f"TimeseriesSequences:\n{np.round(df,4)}\n")
        return df
