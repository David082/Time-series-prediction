# -*- coding: utf-8 -*-
# @author: Longxing Tan, tanlongxing888@163.com
# @date: 2020-03

import sys
import os
filePath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.split(filePath)[0])

import numpy as np
from data.prepare_data import PassengerData
from deepts.model import Model
from config import params


def main():
    x,y=PassengerData(params).get_examples(data_dir='../data/international-airline-passengers.csv')
    print(x.shape,y.shape)

    model=Model(params=params,use_model='seq2seq')
    y_pred=model.predict(x.astype(np.float32), model_dir=params['saved_model_dir'])
    print(y_pred)


if __name__=='__main__':
    main()
