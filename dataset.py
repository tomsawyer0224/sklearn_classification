import sys
if not '.' in sys.path:
    sys.path.append('.')
from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import os

import utils

class ClassificationDataset:
    def __init__(self, dataset: str = 'digit'):
        '''
        args:
            dataset: 'digit' or 'fashion'
        '''
        assert dataset in ['digit', 'fashion'], \
        f'{dataset} is not supported'
        if dataset == 'digit':
            self.dataset = self._get_digit()
        if dataset == 'fashion':
            self.dataset = self._get_fashion()
    def get_dataset(self):
        return self.dataset
    def train_dataset(self):
        return self.dataset['train_data'], self.dataset['train_label']
    def test_dataset(self):
        return self.dataset['test_data'], self.dataset['test_label']
    def id2name(self, label):
        class_name = self.dataset['class_name']
        name = [class_name[l] for l in label]
        return name
    @property
    def name(self):
        return self.dataset['name']
    @property
    def class_name(self):
        return self.dataset['class_name']
    def visualize(self, save_dir: str, n_samples: int = 16):
        '''
        args:
            save_dir: path/to/save
        '''
        os.makedirs(save_dir, exist_ok = True)

        n_train = len(self.dataset['train_data'])
        train_indices = np.random.choice(n_train, size = (n_samples,))
        train_sample_img = self.dataset['train_data'][train_indices]
        train_sample_lbl = self.dataset['train_label'][train_indices]
        train_cls_name = self.id2name(train_sample_lbl)
        train_path = os.path.join(
            save_dir,
            f'{self.dataset["name"]}_train_samples.png'
        )
        utils.save_image(
            images = train_sample_img, 
            path = train_path,
            titles = train_cls_name,
            figsize = (10, 10)
        )

        n_test = len(self.dataset['test_data'])
        test_indices = np.random.choice(n_test, size = (n_samples,))
        test_sample_img = self.dataset['test_data'][test_indices]
        test_sample_lbl = self.dataset['test_label'][test_indices]
        test_cls_name = self.id2name(test_sample_lbl)
        test_path = os.path.join(
            save_dir,
            f'{self.dataset["name"]}_test_samples.png'
        )
        utils.save_image(
            images = test_sample_img, 
            path = test_path,
            titles = test_cls_name,
            figsize = (10, 10)
        )
    def _get_digit(self):
        digit = datasets.load_digits()
        images = digit.images
        labels = digit.target
        class_name = [str(i) for i in range(10)]
        '''
        class_name = {
            k: str(v) for k, v in enumerate(range(10))
        }
        '''
        X_train, X_test, y_train, y_test = train_test_split(
            images,
            labels,
            test_size = 0.3,
            random_state = 42,
            shuffle = False
        )
        ds = {
            'train_data': X_train,
            'train_label': y_train,
            'test_data': X_test,
            'test_label': y_test,
            'class_name': class_name,
            'name': 'digit'
        }
        return ds
    def _get_fashion(self):
        fashion = tf.keras.datasets.fashion_mnist.load_data()
        (X_train, y_train), (X_test, y_test) = fashion
        class_name = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
        '''
        class_name = {
            k: v for k, v in enumerate(class_name)
        }
        '''
        ds = {
            'train_data': X_train,
            'train_label': y_train,
            'test_data': X_test,
            'test_label': y_test,
            'class_name': class_name,
            'name': 'fashion'
        }
        return ds





