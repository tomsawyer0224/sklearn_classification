import sys

if not "." in sys.path:
    sys.path.append(".")
import yaml
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dataset
import utils


class TrainingClassifier:
    def __init__(self, config_file):
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        self.config = config

    def run(self):
        # check if using hyper-param tuning or not
        if "search" in self.config["general_config"]:
            search = True
        else:
            search = False

        # model
        classifier = utils.get_model(self.config, normalize=True)
        classifier_name = self.config["general_config"]["model"]
        # dataset
        cls_dataset = dataset.ClassificationDataset(
            self.config["general_config"]["dataset"]
        )
        train_data, train_label = cls_dataset.train_dataset()
        test_data, test_label = cls_dataset.test_dataset()
        class_name = cls_dataset.class_name
        dataset_name = cls_dataset.name

        n_train = len(train_data)
        n_test = len(test_data)

        # log dir
        root_log_dir = self.config["general_config"]["root_log_dir"]
        model_name = self.config["general_config"]["model"]
        log_dir = utils.make_log_dir(root_log_dir, model_name)
        images_dir = os.path.join(log_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        texts_dir = os.path.join(log_dir, "texts")
        os.makedirs(texts_dir, exist_ok=True)

        # visualize dataset
        cls_dataset.visualize(save_dir=images_dir, n_samples=16)

        # train
        classifier.fit(train_data.reshape(n_train, -1), train_label)

        # test
        predicted_label = classifier.predict(test_data.reshape(n_test, -1))

        # report
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.suptitle(f"{classifier_name} on {dataset_name} dataset")
        utils.save_confusion_matrix(
            y_true=test_label,
            y_pred=predicted_label,
            class_name=class_name,
            save_path=os.path.join(images_dir, "confusion_matrix.png"),
            normalize="true",
            xticks_rotation=45,
            values_format=".2f",
            ax=ax,
        )
        utils.save_classification_report(
            y_true=test_label,
            y_pred=predicted_label,
            class_name=class_name,
            save_dir=texts_dir,
        )
        if search:
            utils.save_best_config(
                classifier, os.path.join(texts_dir, "best_config.yaml")
            )

        # visualize some predictions
        n = min(n_test, 16)
        indices = np.random.choice(n_test, size=(n,))
        imgs = test_data[indices]
        gt_id = test_label[indices]  # ground truth
        gt_name = cls_dataset.id2name(gt_id)
        pr_id = predicted_label[indices]  # predicted
        pr_name = cls_dataset.id2name(pr_id)
        titles = [f"pr:{pr}/gt:{gt}" for pr, gt in zip(pr_name, gt_name)]
        save_path = os.path.join(images_dir, f"{dataset_name}_predicted.png")
        utils.save_image(images=imgs, path=save_path, titles=titles, figsize=(10, 10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    args = parser.parse_args()

    training = TrainingClassifier(args.config_file)
    training.run()
