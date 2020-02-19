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

from train_and_eval import train_and_eval
from predict import predict

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
    ap.add_argument('-a', '--assets', required=True, help='종목1, 종목2, 종목3')
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
    ap.add_argument('--iter', type=int, default=10, help='# of iteration: 10*')
    ap.add_argument('--wavelet', '--no-wavelet', dest='wavelet', default=True, action=BooleanAction, help='wavelet transform 실행 여부: T*')
    ap.add_argument('--train', '--no-train', dest='train', default=False, action=BooleanAction, help='training 실행 여부')
    ap.add_argument('--predict', '--no-predict', dest='predict', default=False, action=BooleanAction, help='prediction 실행 여부')
    ap.add_argument('--save', '--no-save', dest='save', default=False, action=BooleanAction, help='whether or not to save a model: F*')
    ap.add_argument('--pic', '--no-pic', dest='pic', default=False, action=BooleanAction, help='whether or not to show a graph: F*')
    ap.add_argument('--debug', '--no-debug', dest='debug', default=False, action=BooleanAction, help='디버거 사용 유무')
    ap.add_argument('-v', '--verbose', type=int, default=0, help='verbose level: 0*')

    args = vars(ap.parse_args())
    if args['verbose']:
        logger.setLevel(logging.DEBUG)

    logger.info(f"Argument: \n{args}\n")

    assets = [x.strip() for x in args.get('assets').split(",")]
    for asset in assets:
        args['asset'] = asset
        # pdb.set_trace()
        if args.get('train'):
            train_and_eval(args, logger=logger)

        # pdb.set_trace()
        if args.get('predict'):
            args['model'] = f'./models/{asset}.hdf5'
            predict(args, logger=logger)
