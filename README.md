# This is a personal project, for educational purposes only!
# About this project:
  1. This project provides an easy way to use many classification models in the scikit-learn library.
  2. You can train, tune the hyper-parameters, validate any model by using only one command line.
  3. Supports two datasets (hand-written, fashion mnist) and almost classification models (./configs/train)
# How to use:
  1. Clone project, cd to sklearn_classification.
  2. Install the requirements: pip install -q -r requirements.txt
  3. To train the model: modify the config file (i.e './configs/train/RidgeClassifier.yaml') then run the bellow command:
     ```
     python train.py --config_file './configs/train/RidgeClassifier.yaml'
     ```
     see the "./resuls/RidgeClassifier" folder for more details.
     ![image](https://github.com/user-attachments/assets/29ba3631-1b33-419c-9ff2-8477237159aa) \
     ![image](https://github.com/user-attachments/assets/5b9b9313-4a0b-4b37-b523-463ec9684306) \
     ```
                       precision    recall  f1-score   support
    
     T-shirt/top       0.78      0.80      0.79      1000
         Trouser       0.97      0.95      0.96      1000
        Pullover       0.71      0.69      0.70      1000
           Dress       0.79      0.85      0.82      1000
            Coat       0.68      0.75      0.71      1000
          Sandal       0.84      0.84      0.84      1000
           Shirt       0.67      0.46      0.55      1000
         Sneaker       0.86      0.91      0.89      1000
             Bag       0.89      0.93      0.91      1000
      Ankle boot       0.89      0.93      0.91      1000
    
        accuracy                           0.81     10000
       macro avg       0.81      0.81      0.81     10000
    weighted avg       0.81      0.81      0.81     10000
```


