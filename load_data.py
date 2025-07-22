import logging
import pickle

import numpy as np

class DataLoader:
    @staticmethod
    def unpickle(file):
        with open (file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    @staticmethod
    def load_data(train_data_paths, test_data_path):
        test_data = DataLoader.unpickle(test_data_path)
        train_data_dicts = []

        for path in train_data_paths:
            train_data_dicts.append(DataLoader.unpickle(path))
            logging.info(f'batch with path={path} unpickled')

        X_train = np.concatenate([d[b'data'] for d in train_data_dicts], axis=0)
        y_train = sum([d[b'labels'] for d in train_data_dicts], [])
        X_test, y_test = test_data[b'data'], test_data[b'labels']

        return X_train, y_train, X_test, y_test
