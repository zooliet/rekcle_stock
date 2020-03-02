#!/usr/bin/env python

import numpy as np
# import sys
# sys.path.insert(0, '.')

# debugger 셋팅
import pdb
import rlcompleter
pdb.Pdb.complete=rlcompleter.Completer(locals()).complete

# logger 셋팅
import json
import logging
import logging.config

from libs.loaders import StockDatasetLoader
from libs.preprocessing import LabelGenerator
from libs.preprocessing import FeatureSelector
from libs.preprocessing import WaveletTransform
from libs.preprocessing import Scaler
from libs.preprocessing import TimeseriesSequence
from libs.preprocessing import SimpleClassifier
from libs.preprocessing import ColumnRemover

from libs.nn import RekcleLSTM
from libs.nn import RekcleTCN
from libs.nn import RekcleSTCN

from sklearn.externals import joblib

def train_and_eval(args, logger):
    # Dataset 로딩
    dl = StockDatasetLoader(asset=args['asset'], path='dataset/tai', logger=logger)
    df = dl.load(from_date=args['from']) # df: data_frame

    # Y-Label 생성
    # lg = LabelGenerator(logger=logger)
    # df = lg.process(df, target='Close')

    # Technical indicator 선택
    fs = FeatureSelector(logger=logger)
    df = fs.process(df, selector=args['preset'])

    # wavelet transform 수행
    if args['wavelet']:
        wt = WaveletTransform(wavelet_type='db8', excluding=['Close'], logger=logger)
        df_cA = wt.process(df, cA_scale_factor=1, cD_scale_factor=0)
        df_cD = wt.process(df, cA_scale_factor=0, cD_scale_factor=1)
        dfs = [df_cA, df_cD]
    else:
        dfs = [df]

    # train과 test 데이터로 구분 (날짜로 구분)
    dfs = list(map(lambda df: dl.split_by_date(df, from_date=args['from'], to_date=args['to']), dfs))
    df_trains = list(zip(*dfs))[0]
    df_tests = list(zip(*dfs))[1]

    # Scaling 수행
    df_zipped = zip(df_trains, df_tests)
    df_trains = []
    df_tests = []
    scalers = []
    for df_train, df_test in df_zipped:
        scs = [Scaler(op='std', excluding=['Close'], logger=logger), Scaler(op='maxabs', logger=logger)]
        scaler = []
        for sc in scs:
            df_train = sc.process(df_train)
            df_test = sc.process(df_test, scaler=sc.scaler)
            scaler.append(sc.scaler)

        df_trains.append(df_train)
        df_tests.append(df_test)
        scalers.append(scaler)

    # scalers 저장
    asset = args.get('asset')
    arch = args.get('arch')
    scaler_path = f"./models/{asset}_{arch}.scaler.pkl"
    joblib.dump(scalers, scaler_path)

    # Timeseries 시퀀스로 재정렬
    ts = TimeseriesSequence(args['step_in'], args['step_out'], 0, target='Close', logger=logger)
    df_trains = list(map(lambda df_train: ts.sequence_generate(df_train), df_trains))
    df_tests = list(map(lambda df_test: ts.sequence_generate(df_test), df_tests))

    # Make Y label
    cls = SimpleClassifier(['^Close[+]'], logger=logger)
    df_trains = list(map(lambda df_train: cls.process(df_train), df_trains))
    df_tests = list(map(lambda df_test: cls.process(df_test), df_tests))

    # 불필요한 columns 삭제
    cr = ColumnRemover(['^Close[-+]|^Close$'], logger=logger)
    df_trains = list(map(lambda df_train: cr.process(df_train), df_trains))
    df_tests = list(map(lambda df_test: cr.process(df_test), df_tests))

    # Deep Learning ...

    # X_train, y_train, X_test, y_test 추출
    X_train = list(map(lambda df_train: df_train.values[:, :-1], df_trains))
    X_test = list(map(lambda df_test: df_test.values[:, :-1], df_tests))
    y_train = df_trains[0].values[:, -1:]
    y_test = df_tests[0].values[:, -1:]

    # Deep Learning 에서 필요한 파라미터 추가
    args['num_classes'] = cls.num_classes # 2
    args['num_features'] = X_train[0].shape[1] // args['step_in']
    logger.info(f"args: \n{args}\n")

    models = {
        'lstm': RekcleLSTM,
        # 'tcn': RekcleTCN,
        # 'stcn': RekcleSTCN,
        # 'xgb': RekcleXGBModel,
    }

    model = models[args['arch']](logger=logger)
    model.build(**args)

    if args.get('train'):
        acc_threshold = 0.2
        num_iter = args.get('iter', 10)
        for i in range(num_iter):
            logger.info(f"[{args.get('asset')}] Iteration starts... ({i+1}/{num_iter})")
            H = model.fit(X=X_train, y=y_train, iter=i+1)
            loss, acc =  model.evaluate(X=X_test, y=y_test)

            if args.get('save'):
                if acc > acc_threshold:
                    acc_threshold = acc
                    model.save(iter=i+1)

    if args['debug']:
        from keras import backend as K
        # pdb.set_trace()
        # import code
        # code.interact(local=locals())
        # model.args['epoch_at'] += model.args['epochs']
        # lr =  K.get_value(model.optimizer.lr)
        # K.set_value(model.optimizer.lr, lr/10)
        # model.fit(X_train, y_train)

if __name__ == '__main__':
    with open('./config/logging.conf.json', 'rt') as f:
        config = json.load(f)

    logging.config.dictConfig(config)
    logger = logging.getLogger(__name__)
    logger.info("Started...")

    # CLI 파싱
    import argparse
    from libs.utils import BooleanAction

    ap = argparse.ArgumentParser()
    ap.add_argument('-a', '--asset', required=True, help='종목 코드')
    ap.add_argument('--model', default='', help='path to model file')
    ap.add_argument('--arch', default='lstm', help='lstm*, tcn, stcn, convlstm, cnnlstm, cnn, xgb')
    ap.add_argument('--from', default='2010-01-01', help='training 시작일: 2010-01-01*')
    ap.add_argument('--to', default='2018-12-31', help='training 종료일: 2018-12-31*')
    ap.add_argument('--date', default='latest', help='주가 예측일')
    ap.add_argument('-p', '--preset', default=2, type=int, help='feature preset: 2*')
    ap.add_argument('-i', '--step_in', default=6, type=int, help='# of step-in: 6*')
    ap.add_argument('-o', '--step_out', default=1, type=int, help='# of step-out: 1*')
    ap.add_argument('-e', '--epochs', type=int, default=100, help='# of epochs: 100*')
    ap.add_argument('-b', '--batch_size', type=int, default=32, help='# of batch_size: 32*')
    ap.add_argument('-l', '--lr', type=float, default=0.001, help='learning rate: 0.001*')
    ap.add_argument('--iter', type=int, default=1, help='# of iteration: 1*')
    ap.add_argument('--wavelet', '--no-wavelet', dest='wavelet', default=True, action=BooleanAction, help='wavelet transform 실행 여부: T*')
    ap.add_argument('--train', '--no-train', dest='train', default=False, action=BooleanAction, help='training 실행 여부')
    # ap.add_argument('--predict', '--no-predict', dest='predict', default=False, action=BooleanAction, help='prediction 실행 여부')
    ap.add_argument('--save', '--no-save', dest='save', default=False, action=BooleanAction, help='whether or not to save a model: F*')
    ap.add_argument('--pic', '--no-pic', dest='pic', default=False, action=BooleanAction, help='whether or not to show a graph: F*')
    ap.add_argument('--debug', '--no-debug', dest='debug', default=False, action=BooleanAction, help='디버거 사용 유무')
    ap.add_argument('-v', '--verbose', type=int, default=0, help='verbose level: 0*')

    args = vars(ap.parse_args())
    if args['verbose']:
        logger.setLevel(logging.DEBUG)

    if args['debug']:
        pdb.set_trace()

    logger.info(f"Argument: \n{args}\n")
    train_and_eval(args, logger=logger)
