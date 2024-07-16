import sys
if not '.' in sys.path:
    sys.path.append('.')
import unittest

import dataset

class Test_datasets(unittest.TestCase):
    def setUp(self):
        print('---setUp---')
        self.digit = dataset.ClassificationDataset('digit')
        print(
            f'digit.name = {self.digit.name}, digit.class_name = {self.digit.class_name}'
        )
        self.fashion = dataset.ClassificationDataset('fashion')
        print(
            f'fashion.name = {self.fashion.name}, fashion.class_name = {self.fashion.class_name}'
        )
        print('-'*30)
    def get_info(self, arr):
        return f'shape = {arr.shape}, min = {arr.min()}, max = {arr.max()}'
    def test_digit(self):
        print('---test_digit---')
        for k, v in self.digit.get_dataset().items():
            if k not in ['class_name', 'name']:
                print(f'{k}: {self.get_info(v)}')
            else:
                print(f'{k}: {v}')
        print('+'*20)
        train_data, train_label = self.digit.train_dataset()
        print(f'train_data: {self.get_info(train_data)}')
        print(f'train_label: {self.get_info(train_label)}')
        print('+'*20)
        test_data, test_label = self.digit.test_dataset()
        print(f'test_data: {self.get_info(test_data)}')
        print(f'test_label: {self.get_info(test_label)}')
        print('-'*30)
    def test_fashion(self):
        print('---test_digit---')
        for k, v in self.fashion.get_dataset().items():
            if k not in ['class_name', 'name']:
                print(f'{k}: {self.get_info(v)}')
            else:
                print(f'{k}: {v}')
        print('+'*20)
        train_data, train_label = self.fashion.train_dataset()
        print(f'train_data: {self.get_info(train_data)}')
        print(f'train_label: {self.get_info(train_label)}')
        print('+'*20)
        test_data, test_label = self.fashion.test_dataset()
        print(f'test_data: {self.get_info(test_data)}')
        print(f'test_label: {self.get_info(test_label)}')
        print('-'*30)
    def test_visualize(self):
        print('---test_visualize---')
        self.digit.visualize('./tests/test_result', n_samples = 16)
        self.fashion.visualize('./tests/test_result', n_samples = 16)
        print('-'*30)
if __name__ == "__main__":
    unittest.main()
        
