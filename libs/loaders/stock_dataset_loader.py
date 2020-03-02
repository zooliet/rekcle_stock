import numpy as np
import pandas as pd
import logging
from os import path

class StockDatasetLoader:
    def __init__(self, asset, path, logger=None):
        self.asset = asset
        self.path = f'{path}/{asset}_tai.csv'
        # self.key = os.environ['API_KEY']
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

    def load(self, from_date=None, features=None):
        if not path.exists(self.path):
            self.logger.error('데이터 파일이 존재하지 않습니다.')
            exit(code=0)
        else:
            df = pd.read_csv(self.path, index_col=1) # Date 컬럼을 index로 지정
            df.drop(['Unnamed: 0'], inplace=True, axis=1) # Unnamed: 0 란 칼럼을 삭제
            df = df.sort_index() # 시간순으로 배열

        if from_date:
            df = df[from_date:]

        self.logger.debug("StockDataset: \n{}\n".format(np.round(df,4)))
        # self.pristine = df.copy()
        return df

    def split_by_date(self, df, from_date, to_date):
        df_train = df[from_date:to_date]
        df_test = df[df.index > to_date]
        self.logger.debug("Training from {} to {}".format(df_train.index[0], df_train.index[-1]))
        self.logger.debug("Testing from {} to {}".format(df_test.index[0], df_test.index[-1]))
        return df_train, df_test

    def split_by_ratio(self, df, ratio):
        idx = int(len(df) * ratio)
        self.logger.info("Splitting at: {}".format(idx))

        df_train = df[:idx]
        df_test = df[idx:]

        self.logger.debug("Training from {} to {}".format(df_train.index[0], df_train.index[-1]))
        return df_train, df_test

    def find_split_index(self, date, step_in, step_out):
        # num_trains = self.pristine.loc[:date].shape[0] # date를 포함한 갯수
        num_trains = self.pristine.index.get_loc(date) + 1 # date를 포함한 갯수
        # self.split_index = num_trains - step_in + 1
        self.split_index = num_trains
        self.logger.info("Splitting at: {}".format(self.split_index))
