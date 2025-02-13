import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pickle import dump, load
import yaml
from scipy.stats import uniform, norm, randint

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import (
    RidgeClassifier,
    LogisticRegression,
    SGDClassifier,
    Perceptron,
    PassiveAggressiveClassifier,
)
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    RandomizedSearchCV,
    HalvingRandomSearchCV,
)

from sklearn.metrics import ConfusionMatrixDisplay, classification_report


def save_image(
    images: np.ndarray | list[np.ndarray],
    path: str,
    titles: list[str] | np.ndarray = None,
    figsize=(10, 10),
) -> None:
    """
    args:
        images: array of shape (n, h, w)
        path: path to save images
        title: title of images
    """
    n_samples = len(images)
    if titles is not None:
        assert len(images) == len(titles)
    else:
        titles = [""] * n_samples
    # r = c = int(np.ceil(np.sqrt(n_samples)))
    n_samples = min(16, n_samples)
    r = c = int(np.sqrt(n_samples))
    fig, axes = plt.subplots(r, c, figsize=figsize)
    idx = 0
    for i in range(r):
        for j in range(c):
            axes[i, j].imshow(images[idx], cmap=plt.cm.gray_r, interpolation="nearest")
            axes[i, j].axis("off")
            axes[i, j].set_title(titles[idx])
            idx += 1
    # fig.savefig(path, bbox_inches='tight')
    fig.savefig(path)
    plt.close()


def _adjust_param_distributions(param_distributions: dict):
    """
    converts string to expression
    """
    supported_distributions = ["uniform", "norm", "randint"]
    for k, v in param_distributions.items():
        if isinstance(v, str):
            check = sum([True if d in v else False for d in supported_distributions])
            if check > 0:
                param_distributions[k] = eval(v)


def get_model(config: dict, normalize: bool = True):
    """
    args:
        config: dict-like parameters
        normalize: normalize the input or not
    returns:
        sklearn model
    """
    # model_config = {k: v for k, v in config.items() if k != 'model'}
    # classifier = eval(config['model'])(**model_config)
    if config["model_config"] is None:
        config["model_config"] = {}
    classifier = eval(config["general_config"]["model"])(**config["model_config"])
    if "search" in config["general_config"].keys():
        search_config = {"estimator": classifier} | config["search_config"]
        search_config["refit"] = True
        if "param_distributions" in search_config.keys():
            _adjust_param_distributions(
                search_config["param_distributions"],
            )
        classifier = eval(config["general_config"]["search"])(**search_config)
    if normalize:
        model = make_pipeline(StandardScaler(), classifier)
    else:
        model = classifier
    return model


def make_log_dir(root_log_dir, model_name=None):
    if model_name is not None:
        root = os.path.join(root_log_dir, model_name)
    else:
        root = root_log_dir
    os.makedirs(root, exist_ok=True)
    folders = []
    for item in os.listdir(root):
        if os.path.isdir(os.path.join(root, item)):
            folders.append(item)
    ver = 0
    for folder in folders:
        if "version" in folder:
            ver += 1
    log_dir = os.path.join(root, f"version_{ver}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_name: list | None = None,
    save_path: str = None,
    **kwargs,
):
    """
    args:
        y_true: 1-D array of ground truth labels
        y_pred: 1-D array of predicted labels
        class_name: list-like of shape (n_classes, )
        save_path: path/to/save/image.png
        kwargs: keyword arguments of the ConfusionMatrixDisplay.from_predictions() method
    """
    ConfusionMatrixDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, display_labels=class_name, **kwargs
    )
    if save_path is None:
        save_path = "./confusion_matrix.png"
    plt.savefig(save_path)
    plt.close()


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_name: list | None = None,
    save_dir: str = None,
    **kwargs,
):
    """
    args:
        y_true: 1-D array of ground truth labels
        y_pred: 1-D array of predicted labels
        class_name: list-like of shape (n_classes, )
        save_dir: path/to/save
        kwargs: keyword arguments of the classification_report method
    """
    os.makedirs(save_dir, exist_ok=True)
    cls_rpt_str = classification_report(
        y_true=y_true, y_pred=y_pred, target_names=class_name, output_dict=False
    )
    with open(os.path.join(save_dir, "cls_rpt.txt"), "w") as text_file:
        text_file.write(cls_rpt_str)

    cls_rpt_dict = classification_report(
        y_true=y_true, y_pred=y_pred, target_names=class_name, output_dict=True
    )
    with open(os.path.join(save_dir, "cls_rpt.json"), "w") as json_file:
        json.dump(cls_rpt_dict, json_file)


def save_checkpoint(model, save_path):
    with open(save_path, "wb") as f:
        dump(model, f, protocol=5)


def load_checkpoint(checkpoint_path):
    with open(checkpoint_path, "rb") as f:
        model = load(f)
    return model


def save_best_config(model, save_path):
    """
    args:
        model: instance of ...SearchCV
        save_path: path/to/config.yaml
    """
    if isinstance(model, Pipeline):
        clf = list(model)[-1]
    else:
        clf = model
    # params = clf.best_estimator_.get_params()
    params = clf.best_params_

    # convert numpy value to python float or int to save to yaml file
    for k, v in params.items():
        if isinstance(v, (np.float32, np.float64)):
            params[k] = float(v)
        if isinstance(v, (np.int32, np.int64)):
            params[k] = int(v)
    config = {"model_config": params}
    with open(save_path, "w") as f:
        yaml.dump(config, f)
