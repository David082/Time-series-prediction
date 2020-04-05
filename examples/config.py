
import json
import argparse
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument('--use_model',type=str, default='seq2seq',help='use model for train, seq2seq, tcn, transformer')
parser.add_argument('--data_dir',type=str, default='../data/international-airline-passengers.csv',help='dataset directory')
parser.add_argument('--model_dir',type=str, default='../models/checkpoint',help='saved checkpoint directory')
parser.add_argument('--saved_model_dir',type=str, default='../models',help='saved pb directory')
parser.add_argument('--input_seq_length',type=int,default=16,help='sequence length for input')
parser.add_argument('--output_seq_length',type=int,default=4,help='sequence length for output')
parser.add_argument('--n_epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for training')

args = parser.parse_args()
params = vars(args)


class Config(object):
    def __init__(self):
        self.params = defaultdict()

    def from_json_file(self, json_file):
        with open(json_file, 'r') as f:
            self.params = json.load(f)

    def to_json_string(self, json_file, params):
        with open(json_file, 'w') as f:
            json.dump(params, f)


if __name__ == '__main__':
    config = Config()
    config.to_json_string('./config.json', params)
    #config.from_json_file('./config.json')
