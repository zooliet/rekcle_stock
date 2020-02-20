
import pdb
import rlcompleter
pdb.Pdb.complete=rlcompleter.Completer(locals()).complete

import time
import numpy as np

from keras.models import load_model
from sklearn.model_selection import train_test_split

from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, Flatten, Dropout
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

from libs.callbacks import TrainingMonitor
from libs.callbacks import EpochCheckpoint
from keras.callbacks import ModelCheckpoint

class RekcleLSTM:
    def __init__(self, logger=None):
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

    def build(self, **args):
        self.args = args
        model_path = args.get('model', '')
        num_features = args.get('num_features', 1)
        num_classes = args.get('num_classes', 2)
        step_in = args.get('step_in', 5)
        step_out = args.get('step_out', 1)
        lr = args.get('lr', 0.001)

        if model_path:
            try:
                self.inner_model = load_model(model_path)
            except:
                debug.error("모델 파일이 존재하지 않습니다.")
                exit(code=0)
        else:
            input_cA = Input(shape=(step_in, num_features))
            x_cA = LSTM(units=128,
                     activation='sigmoid',
                     recurrent_activation='hard_sigmoid',
                     use_bias=True,
                     kernel_initializer='glorot_uniform',
                     recurrent_initializer='orthogonal',
                     bias_initializer='zeros',
                     unit_forget_bias=True,
                     kernel_regularizer=None,
                     recurrent_regularizer=None,
                     bias_regularizer=None,
                     activity_regularizer=None,
                     kernel_constraint=None,
                     recurrent_constraint=None,
                     bias_constraint=None,
                     dropout=0.0,
                     recurrent_dropout=0.0,
                     implementation=1,
                     return_sequences=True,
                     return_state=False,
                     go_backwards=False,
                     stateful=False,
                     unroll=False)(input_cA)
            x_cA = Dropout(0.2)(x_cA)
            x_cA = LSTM(units=128,
                     activation='sigmoid',
                     recurrent_activation='hard_sigmoid',
                     use_bias=True,
                     kernel_initializer='glorot_uniform',
                     recurrent_initializer='orthogonal',
                     bias_initializer='zeros',
                     unit_forget_bias=True,
                     kernel_regularizer=None,
                     recurrent_regularizer=None,
                     bias_regularizer=None,
                     activity_regularizer=None,
                     kernel_constraint=None,
                     recurrent_constraint=None,
                     bias_constraint=None,
                     dropout=0.0,
                     recurrent_dropout=0.0,
                     implementation=1,
                     return_sequences=False,
                     return_state=False,
                     go_backwards=False,
                     stateful=False,
                     unroll=False)(x_cA)
            x_cA = Dropout(0.2)(x_cA)

            input_cD = Input(shape=(step_in, num_features))
            x_cD = LSTM(units=128,
                     activation='sigmoid',
                     recurrent_activation='hard_sigmoid',
                     use_bias=True,
                     kernel_initializer='glorot_uniform',
                     recurrent_initializer='orthogonal',
                     bias_initializer='zeros',
                     unit_forget_bias=True,
                     kernel_regularizer=None,
                     recurrent_regularizer=None,
                     bias_regularizer=None,
                     activity_regularizer=None,
                     kernel_constraint=None,
                     recurrent_constraint=None,
                     bias_constraint=None,
                     dropout=0.0,
                     recurrent_dropout=0.0,
                     implementation=1,
                     return_sequences=True,
                     return_state=False,
                     go_backwards=False,
                     stateful=False,
                     unroll=False)(input_cD)
            x_cD = Dropout(0.2)(x_cD)
            x_cD = LSTM(units=128,
                     activation='sigmoid',
                     recurrent_activation='hard_sigmoid',
                     use_bias=True,
                     kernel_initializer='glorot_uniform',
                     recurrent_initializer='orthogonal',
                     bias_initializer='zeros',
                     unit_forget_bias=True,
                     kernel_regularizer=None,
                     recurrent_regularizer=None,
                     bias_regularizer=None,
                     activity_regularizer=None,
                     kernel_constraint=None,
                     recurrent_constraint=None,
                     bias_constraint=None,
                     dropout=0.0,
                     recurrent_dropout=0.0,
                     implementation=1,
                     return_sequences=False,
                     return_state=False,
                     go_backwards=False,
                     stateful=False,
                     unroll=False)(x_cD)
            x_cD = Dropout(0.2)(x_cD)

            x = concatenate([x_cA, x_cD])
            output = Dense(num_classes)(x)
            model = Model(inputs=[input_cA, input_cD], outputs=output)
            optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            # model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
            self.inner_model = model

        if args.get('verbose', 0):
            model.summary()

    def fit(self, X, y, iter=0):
        num_classes = self.args.get('num_classes', 2)
        num_features = self.args.get('num_features', 1)
        step_in = self.args.get('step_in', 5)
        epoch_at = self.args.get('epoch_at', 0)
        asset = self.args.get('asset', 'Test')

        # validation 데이터
        X, X_valid, y, y_valid = train_test_split(np.hstack(X), y, test_size=0.2, shuffle=True)
        num_of_half_columns = X.shape[1]//2
        X = [X[:, :num_of_half_columns], X[:, num_of_half_columns:]]
        X_valid = [X_valid[:, :num_of_half_columns], X_valid[:, num_of_half_columns:]]

        # input shape 조정
        X = list(map(lambda x: x.reshape((-1, step_in, num_features)), X))
        X_valid = list(map(lambda x: x.reshape((-1, step_in, num_features)), X_valid))

        y = np_utils.to_categorical(y, num_classes)
        y_valid = np_utils.to_categorical(y_valid, num_classes)

        callbacks = [TrainingMonitor(epoch_at=epoch_at, output_path=f'./plots/{asset}_{iter:03d}', logger=self.logger)]
        if self.args['save']:
            pass
            # model_path = f"./tmp/{asset}_{iter:03d}"
            # callbacks.append(EpochCheckpoint(epoch_at=epoch_at, model_path=f'./tmp/{model_path}', every=5, logger=self.logger))
            # or
            # # model_path = f"./tmp/{asset}_"+"{epoch:03d}-{val_loss:.4f}.hdf5"
            # model_path = f"./tmp/{asset}_{iter:03d}"+"-{val_loss:.4f}.hdf5"
            # callbacks.append(ModelCheckpoint(model_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1))

        params = {
            'epochs': self.args.get('epochs', 100),
            'batch_size': self.args.get('batch_size', 32),
            'verbose': self.args.get('verbose', 1),
            'callbacks': callbacks,
        }
        tic = time.time()
        H = self.inner_model.fit(X, y, validation_data=(X_valid, y_valid), **params)
        toc = time.time()
        self.logger.info("Training duration: {:.4f} sec".format((toc-tic)))
        return H

    def evaluate(self, X, y):
        num_classes = self.args.get('num_classes', 2)
        num_features = self.args.get('num_features', 1)
        step_in = self.args.get('step_in', 5)

        # input shape 조정
        X = list(map(lambda x: x.reshape((-1, step_in, num_features)), X))
        y = np_utils.to_categorical(y, num_classes)

        params = {
            'batch_size': self.args.get('batch_size', 32),
            'verbose': self.args.get('verbose', 1),
        }
        loss, acc = self.inner_model.evaluate(X, y, **params)
        return (loss, acc)

    def predict(self, X=[]):
        num_classes = self.args.get('num_classes', 2)
        num_features = self.args.get('num_features', 1)
        step_in = self.args.get('step_in', 5)

        X = list(map(lambda x: x.reshape((-1, step_in, num_features)), X))
        params = {
            'verbose': self.args.get('verbose', 1),
        }
        pred = self.inner_model.predict(X, **params)
        # self.logger.debug("Prediction: {}".format(pred))
        pred = np.argmax(pred, axis=-1)
        return pred

    def save(self, iter=0):
        asset = self.args.get('asset', 'Test')
        arch = self.args.get('arch', 'arch')
        # path = f"./models/{asset}_{iter:03d}.hdf5"
        model_path = f"./models/{asset}_{arch}.model.hdf5"
        self.logger.info(f'Saving a model to {model_path}')
        self.inner_model.save(model_path, overwrite=True)
