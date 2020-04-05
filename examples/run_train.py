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
    data_loader=DataLoader()
    dataset=data_loader(data_dir=params['data_dir'], batch_size=8,training=True, sample=0.8)
    valid_dataset=data_loader(data_dir=params['data_dir'],batch_size=8, training=True, sample=0.2)

    model=Model(params=params, use_model=params['use_model'], use_loss='mse',use_optimizer='adam')  # model: seq2seq, tcn, transformer
    model.train(dataset,n_epochs=10,mode='eager',export_model=True)  # mode can choose eager or fit
    model.eval(valid_dataset)


if __name__=='__main__':
    main()
