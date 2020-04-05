

import tensorflow as tf
import functools
from data.prepare_data import PassengerData


class DataLoader(object):
    def __init__(self):
        pass

    def __call__(self,data_dir,batch_size,training,sample=1):
        prepare_data = PassengerData(data_dir)

        dataset = tf.data.Dataset.from_tensor_slices(prepare_data.get_examples(data_dir,sample=sample))
        if training:
            dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.batch(batch_size).prefetch(5)
        return dataset

