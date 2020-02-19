import numpy as np
import pandas as pd
import logging

# Constant

class FeatureSelector:
    def __init__(self, logger=None):
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

    def process(self, df, selector):
        FEATURES = {
            0: ['WILL_R','CCI','PLUS_DI','BB_BW','wcl_p','EMA3', 'RSI','CMO','MINUS_DI','OBV','MACD','MOM'],
            1: ['AROONOSC','ROCP','WILL_R','CCI','PLUS_DI','BB_BW','wcl_p','EMA3','RSI','CMO','MINUS_DI','OBV','MACD','MOM','AD','STOCH_K'],
            2: ['ROCP','WILL_R','CCI','PLUS_DI','BB_BW','RSI','CMO','MINUS_DI','OBV','MACD','MOM','AD'],
            3: ['AROONOSC','ROCP','WILL_R','CCI','PLUS_DI','BB_BW', 'RSI','CMO','MINUS_DI','OBV','MACD','MOM','AD','STOCH_K'],
            100: ['AROONOSC','ROCP'],
        }

        features = FEATURES[selector]
        features.append('Close')
        df = df[features]
        self.logger.debug("FeatureSelector: \n{}\n".format(np.round(df,4)))
        return df
