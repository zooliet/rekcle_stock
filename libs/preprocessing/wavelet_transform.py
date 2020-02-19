# http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
# http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/

import numpy as np
import pandas as pd
import logging
import pywt

class WaveletTransform:
    def __init__(self, excluding=['Close'], wavelet_type='db8', logger=None):
        self.excluding = excluding
        self.wavelet_type = wavelet_type
        # http://wavelets.pybytes.com/wavelet
        # 'haar','db4','db6', 'db8', 'db10','db12', 'db14', 'bior2.8', 'bior3.1', 'bior3.3'

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

    def process(self, df, cA_scale_factor, cD_scale_factor):
        columns = df.columns.to_list()
        for target in self.excluding:
            columns.remove(target)

        (cA, cD) = pywt.dwt(df[columns], self.wavelet_type, axis=0)
        (cDA, cDD) = pywt.dwt(cD, self.wavelet_type, axis=0)
        cDAn = pywt.idwt(cDA, np.zeros(cDA.shape), self.wavelet_type, axis=0)
        cDAn = cDAn[:len(cD)]

        cA = cA_scale_factor * cA
        cDAn = cD_scale_factor * cDAn
        result = pywt.idwt(cA, cDAn, self.wavelet_type, axis=0)
        if result.shape[0] != df.shape[0]:
            print(result.shape[0], ' => ', df.shape[0])
        df = df.copy() # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
        df.loc[:, columns] = result[:df.shape[0]]
        self.logger.debug("WaveletTransform: \n{}\n".format(np.round(df,4)))
        return df
