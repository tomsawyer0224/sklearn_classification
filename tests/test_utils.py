import sys

from IPython.lib.display import isfile

if not "." in sys.path:
    sys.path.append(".")
import unittest
import os
import yaml

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import dataset
import utils


class Test_utils(unittest.TestCase):
    """"""

    def setUp(self):
        self.digit = dataset.ClassificationDataset("digit")
        self.fashion = dataset.ClassificationDataset("fashion")

    def test_save_image(self):
        print("---test_save_image---")
        digit_train_data, digit_train_label = self.digit.train_dataset()
        img1 = digit_train_data[100:118]
        lbl1 = digit_train_label[100:118]
        tit1 = self.digit.id2name(lbl1)
        pth1 = "./tests/test_result/digit.png"

        fashion_train_data, fashion_train_label = self.fashion.train_dataset()
        img2 = fashion_train_data[200:220]
        lbl2 = fashion_train_label[200:220]
        tit2 = self.fashion.id2name(lbl2)
        pth2 = "./tests/test_result/fashion.png"

        utils.save_image(img1, pth1, tit1)
        utils.save_image(img2, pth2, tit2)
        print("-" * 30)

    """"""

    def test_get_model(self):
        print("---test_get_model---")
        for f in os.listdir("./configs"):
            if os.path.isfile(os.path.join("./configs", f)) and ".yaml" in f:
                with open(os.path.join("./configs", f), "r") as file:
                    config = yaml.safe_load(file)
                config = config
                clf = utils.get_model(config=config, normalize=True)
                print(clf)
                print("+" * 10)
        print("-" * 30)

    """"""

    def test_make_log_dir(self):
        print("---test_make_log_dir---")
        utils.make_log_dir("./results", "SVC")
        utils.make_log_dir("./results", "LinearSVC")
        utils.make_log_dir("./results", "SVC")
        print("-" * 30)

    def test_save_confusion_matrix(self):
        print("---test_save_confusion_matrix---")
        X, y = make_classification(random_state=0, n_informative=3, n_classes=3)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        clf = SVC(random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        utils.save_confusion_matrix(
            y_true=y_test,
            y_pred=y_pred,
            normalize="true",
            class_name=["cl0", "cl1", "cl2"],
            save_path="./tests/test_result/cm.png",
        )
        print("-" * 30)

        print("---test_save_classification_report---")
        utils.save_classification_report(
            y_true=y_test,
            y_pred=y_pred,
            normalize="true",
            class_name=["cl0", "cl1", "cl2"],
            save_dir="./tests/test_result/logs",
        )
        print("-" * 30)


if __name__ == "__main__":
    unittest.main()
