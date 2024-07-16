import sys
if not '.' in sys.path:
    sys.path.append('.')
import yaml
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import (
    LearningCurveDisplay,
    ValidationCurveDisplay
)

import dataset
import utils

class Validation:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        self.config = self._adjust_params(config)
    def _adjust_params(self, config):
        for k, v in config.items():
            if isinstance(v, dict):
                self._adjust_params(v)
            else:
                if isinstance(v, str) and 'np.' in v:
                    config[k] = eval(v)
        return config
    def run(self, normalize: bool = True):
        '''
        args:
            normalize: whether normalize the input or not
        '''
        # model
        classifier = utils.get_model(self.config, normalize = False)

        # dataset
        cls_dataset = dataset.ClassificationDataset(
            self.config['general_config']['dataset']
        )

        train_data, train_label = cls_dataset.train_dataset()
        #test_data, test_label = cls_dataset.test_dataset()
        #class_name = cls_dataset.class_name
        #dataset_name = cls_dataset.name

        n_train = len(train_data)
        #n_test = len(test_data)

        # log dir
        root_log_dir = self.config['general_config']['root_log_dir']
        model_name = self.config['general_config']['model']
        log_dir = utils.make_log_dir(root_log_dir, model_name)
        #images_dir = os.path.join(log_dir, 'images')
        #os.makedirs(images_dir, exist_ok = True)
        #texts_dir = os.path.join(log_dir, 'texts')
        #os.makedirs(texts_dir, exist_ok = True)

        if normalize:
            scaler = StandardScaler()
            train_data_scaled = scaler.fit_transform(
                train_data.reshape(n_train, -1)
            )
        else:
            train_data_scaled = train_data.reshape(n_train, -1)
        # LearningCurveDisplay
        LearningCurveDisplay.from_estimator(
            estimator = classifier,
            X = train_data_scaled,
            y = train_label,
            **self.config['learning']
        )
        plt.savefig(os.path.join(log_dir, 'learning_curve.png'))
        plt.close()
        # ValidationCurveDisplay
        ValidationCurveDisplay.from_estimator(
            estimator = classifier,
            X = train_data_scaled,
            y = train_label,
            **self.config['validation']
        )
        plt.savefig(os.path.join(log_dir, 'validation_curve.png'))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type = str)
    parser.add_argument('--normalize', type = bool, default = True)
    args = parser.parse_args()

    training = Validation(args.config_file)
    training.run(normalize = args.normalize)









