#!/usr/bin/env python

import asyncio
import aioredis
from aiomultiprocess import Worker

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

async def predict(args, logger):
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

    # date 까지의 data_frame을 추출
    if args['date'] != 'latest':
        dfs = list(map(lambda df: dl.split_by_date(df, from_date=args['from'], to_date=args['date']), dfs))
        dfs = list(zip(*dfs))[0]

    # scalers 복원
    asset = args.get('asset')
    arch = args.get('arch')
    scaler_path = f"./models/{asset}_{arch}.scaler.pkl"
    external_scalers = joblib.load(scaler_path)

    # scaling 수행
    zipped = zip(dfs, external_scalers)
    dfs = []
    for df, external_scaler in zipped:
        scs = [Scaler(op='std', excluding=['Close'], logger=logger), Scaler(op='maxabs', logger=logger)]
        for sc, external in zip(scs, external_scaler):
            df = sc.process(df, scaler=external)
        dfs.append(df)

    # Timeseries 시퀀스로 정렬
    ts = TimeseriesSequence(args['step_in'], args['step_out'], 0, target='Close', logger=logger)
    dfs = list(map(lambda df: ts.target_sequence_generate(df), dfs))

    # Make Y label: predict에서는 필요없으나 cls 인스턴스가 필요해서 실행함
    cls = SimpleClassifier(['^Close[+]'], logger=logger)

    # 불필요한 columns 삭제
    cr = ColumnRemover(['^Close[-+]|^Close$'], logger=logger)
    dfs = list(map(lambda df: cr.process(df), dfs))

    # Deep Learning ...

    # X_predict 선정
    X_predict = list(map(lambda df: df.values[-1:,:], dfs))

    # Deep Learning 에서 필요한 파라미터 추가
    args['num_classes'] = cls.num_classes # 2
    args['num_features'] = X_predict[0].shape[1] // args['step_in']
    logger.info(f"args: \n{args}\n")

    models = {
        'lstm': RekcleLSTM,
        # 'tcn': RekcleTCN,
        # 'stcn': RekcleSTCN,
        # 'xgb': RekcleXGBModel,
    }

    model = models[args['arch']](logger=logger)
    model.build(**args)

    # prediction
    pred = model.predict(X=X_predict)
    logger.info(f'Prediction input: \n{np.vstack(X_predict)}\n')
    logger.info(f'Prediction result: {pred[0]}\n')

    if args['debug']:
        from keras import backend as K
        pdb.set_trace()
        # import code
        # code.interact(local=locals())
        # model.args['epoch_at'] += model.args['epochs']
        # lr =  K.get_value(model.optimizer.lr)
        # K.set_value(model.optimizer.lr, lr/10)
        # model.fit(X_train, y_train)


async def main(args, logger):
    redis = await aioredis.create_redis('redis://localhost')
    (ch,) = await redis.subscribe('rekcle:predict')
    while await ch.wait_message():
        asset = await ch.get(encoding='utf-8')
        logger.info(f'{ch.name.decode()}: {asset}')
        args['asset'] = asset
        arch = args.get('arch', 'arch')
        args['model'] = f'./models/{asset}_{arch}.model.hdf5'
        task = asyncio.create_task(predict(args, logger))


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
    # ap.add_argument('-e', '--epochs', type=int, default=100, help='# of epochs: 100*')
    # ap.add_argument('-b', '--batch_size', type=int, default=32, help='# of batch_size: 32*')
    # ap.add_argument('-l', '--lr', type=float, default=0.001, help='learning rate: 0.001*')
    # ap.add_argument('--iter', type=int, default=10, help='# of iteration: 10*')
    ap.add_argument('--wavelet', '--no-wavelet', dest='wavelet', default=True, action=BooleanAction, help='wavelet transform 실행 여부: T*')
    # ap.add_argument('--train', '--no-train', dest='train', default=False, action=BooleanAction, help='training 실행 여부')
    # ap.add_argument('--predict', '--no-predict', dest='predict', default=False, action=BooleanAction, help='prediction 실행 여부')
    # ap.add_argument('--save', '--no-save', dest='save', default=False, action=BooleanAction, help='whether or not to save a model: F*')
    ap.add_argument('--pic', '--no-pic', dest='pic', default=False, action=BooleanAction, help='whether or not to show a graph: F*')
    ap.add_argument('--debug', '--no-debug', dest='debug', default=False, action=BooleanAction, help='디버거 사용 유무')
    ap.add_argument('-v', '--verbose', type=int, default=0, help='verbose level: 0*')

    args = vars(ap.parse_args())
    if args['verbose']:
        logger.setLevel(logging.DEBUG)

    logger.info(f"Argument: \n{args}\n")
    asyncio.run(main(args, logger))
    # predict(args, logger=logger)
