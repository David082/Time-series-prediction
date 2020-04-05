# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-01


import tensorflow as tf
from tensorflow.keras.layers import Input
from deepts.models.seq2seq import Seq2seq
from deepts.models.tcn import TCN
from deepts.models.transformer import Transformer
from deepts.models.unet import Unet
from deepts.models.nbeats import NBeatsNet
from deepts.models.gan import GAN


class Loss(object):
    def __init__(self,use_loss):
        self.use_loss=use_loss

    def __call__(self,):
        if self.use_loss == 'mse':
            return tf.keras.losses.MeanSquaredError()
        elif self.use_loss == 'rmse':
            return tf.math.sqrt(tf.keras.losses.MeanSquaredError())


class Optimizer(object):
    def __init__(self,use_optimizer):
        self.use_optimizer=use_optimizer

    def __call__(self,):
        if self.use_optimizer == 'adam':
            return tf.keras.optimizers.Adam(lr=0.0005)
        elif self.use_optimizer == 'sgd':
            return tf.keras.optimizers.SGD(lr=0.0005)


class Model(object):
    def __init__(self,params, use_model, use_loss='mse',use_optimizer='sgd', custom_model_params=None):
        if use_model == 'seq2seq':
            Model = Seq2seq(custom_model_params)
            inputs = Input([params['input_seq_length'], 1])
            outputs = Model(inputs, training=True, predict_seq_length=params['output_seq_length'])
        elif use_model == 'tcn':
            Model = TCN(custom_model_params)
            inputs = Input([params['input_seq_length'], 1])
            outputs = Model(inputs, training=True, predict_seq_length=params['output_seq_length'])
        elif use_model == 'transformer':
            Model = Transformer(custom_model_params)
            inputs = (Input([16,1]),Input([4,1]))
            outputs = Model(inputs, training=True, predict_seq_length=params['output_seq_length'])
        elif use_model == 'unet':
            Model = Unet(custom_model_params)
            inputs = Input([params['input_seq_length'], 1])
            outputs = Model(inputs, training=True, predict_seq_length=params['output_seq_length'])
        elif use_model == 'nbeats':
            Model = NBeatsNet(custom_model_params)
            inputs = Input([params['input_seq_length']])
            outputs = Model(inputs, training=True, predict_seq_length=params['output_seq_length'])
        else:
            raise ValueError("unsupported use_model of {} yet".format(use_model))

        self.params=params
        self.use_model = use_model
        self.use_loss = use_loss
        self.use_optimizer = use_optimizer
        self.loss_fn = Loss(use_loss)()
        self.optimizer_fn = Optimizer(use_optimizer)()
        self.model = tf.keras.Model(inputs, outputs, name=use_model)

    def train(self, dataset, n_epochs, mode='eager', export_model=False):
        print("-" * 35)
        print("Start to train {}, in {} mode".format(self.use_model, mode))
        print("-" * 35)
        if mode == 'eager':
            self.global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
            for epoch in range(1, n_epochs + 1):
                print("-> EPOCH {}".format(epoch))
                for step, (x, y) in enumerate(dataset.take(-1)):
                    self.train_step(x, y)

        elif mode == 'fit':
            self.model.compile(loss=self.loss_fn, optimizer=self.optimizer_fn)
            callbacks = []
            self.model.fit(dataset,epochs=n_epochs,callbacks=callbacks)
        else:
            print("unsupported train mode of {}, choose 'eager' or 'fit'".format(mode))

        if export_model:
            self.export_model()

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            try:
                y_pred=self.model(tf.cast(x,tf.float32),training=True)
            except:
                y_pred = self.model([tf.cast(x, tf.float32),tf.cast(y,tf.float32)], training=True)
            loss=self.loss_fn(y, y_pred)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer_fn.apply_gradients(zip(gradients, self.model.trainable_variables))
            print("=> STEP %4d  lr: %.6f  loss: %4.2f" % (self.global_steps, self.optimizer_fn.lr.numpy(), loss))
            self.global_steps.assign_add(1)

    def test_step(self, x, y):
        x=tf.cast(x, tf.float32)
        y=tf.cast(y, tf.float32)
        y_pred=self.model(x)
        metrics=self.loss_fn(y, y_pred).numpy()
        return metrics

    def eval(self, valid_dataset):
        for step,(x,y) in enumerate(valid_dataset.take(-1)):
            metrics=self.test_step(x,y)
            print("=> STEP %4d Metrics: %4.2f"%(step, metrics))

    def predict(self, x_test, model_dir, use_model='pb'):
        if use_model=='pb':
            print('Load saved pb model ...')
            model=tf.saved_model.load(model_dir)
        else:
            print('Load checkpoint model ...')
            model=self.model.load_weights(model_dir)

        y_pred=model(tf.constant(x_test),True,None)  # To be clarified
        return y_pred

    def export_model(self):
        '''
        save the model to .pb file for prediction
        :return:
        '''

        tf.saved_model.save(self.model, self.params['saved_model_dir'])
        print("pb_model save in {}".format(self.params['saved_model_dir']))
