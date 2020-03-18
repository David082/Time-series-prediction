# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-01


# categorical_feature
import sys
import os
filePath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.split(filePath)[0])
from data.data_loader import DataLoader
from deepts.model import Model
from config import params


def main():
    data_loader=DataLoader(data_dir=params['data_dir'])
    dataset=data_loader(batch_size=8,training=True)

    model=Model(use_model=params['use_model'], params=params, use_loss='mse',use_optimizer='adam')  # model: seq2seq, tcn, transformer
    model.train(dataset,n_epochs=10,mode='eager')  # mode can choose eager or fit


if __name__=='__main__':
    main()
