'''
Description:
Author: Osvaldo Hernandez-Segura
References: ChatGPT, Pyspark documentation, Numpy documentation, Pandas documentations
'''
import pandas as pd
import numpy as np
import os
import gzip
import argparse
import code_segments
import utils

from joblib import Memory
from transformers import CustomBestFeaturesTransformer, CustomDropNaNColumnsTransformer, CustomClipTransformer, CustomReplaceInfNanWithZeroTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from sklearn.model_selection import GridSearchCV

def underprediction_rate(y_true: pd.DataFrame, y_pred: pd.DataFrame)-> tuple[float, int]:
    '''
    Calculate the underprediction rate and get number of underpredictions.

    :param y_true: the true values.
    y_pred: the predicted values.
    :return: the underprediction rate and number of underpredictions.
    '''
    underpredictions = np.sum(y_pred < y_true)
    total = len(y_true)
    return underpredictions / total, underpredictions

def overprediction_rate(y_true: pd.DataFrame, y_pred: pd.DataFrame)-> tuple[float, int]:  
    '''
    Calculate the overprediction rate and get number of overpredictions.

    :param y_true: the true values.
    :param y_pred: the predicted values.
    :return: the overprediction rate and number of overpredictions.
    '''
    overpredictions = np.sum(y_pred > y_true)
    total = len(y_true)
    return overpredictions / total, overpredictions

def get_classification_pipeline()-> Pipeline:
    '''
    Get the custom classification pipeline.

    :return: the pipeline.
    '''
    mem = Memory(location='cache_dir', verbose=0)
    pipeline = Pipeline(
        steps=[ 
                ("drop_nan", CustomDropNaNColumnsTransformer(threshold=0.6)),
                ("inf_nan", CustomReplaceInfNanWithZeroTransformer()),
                ("clipper", CustomClipTransformer()),
                ("scaler", StandardScaler()),
                ("rfecv", CustomBestFeaturesTransformer()), # comment this out for testing only; must be in final pipeline
                ("classifier", None)
                ],
                memory=mem # cache transformers (to avoid fitting transformers multiple times)
                )
    return pipeline

def run_classification_pipeline(X_train: pd.DataFrame, y_train: pd.DataFrame, X_dev: pd.DataFrame, y_dev: pd.DataFrame, classifiers: list)-> Pipeline:
    '''
    Run the classification pipeline.

    :param data: the data to process.
    :return: the pipeline.
    '''
    pipeline = get_classification_pipeline()

    output_path = os.path.join(os.pardir, "output")
    os.makedirs(output_path, exist_ok=True)
    write_to_path = os.path.join(output_path, "results.txt")
    write_to_file = open(write_to_path, "w", encoding="ascii")

    for idx in range(len(classifiers)):
        pipeline.set_params(classifier=classifiers[idx])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_dev)
        model_name = pipeline.named_steps['classifier'].__class__.__name__
        text = "%s: Accuracy: %.3f\n" % (model_name, accuracy_score(y_dev, predictions)) 
        text += "%s: AUC: %.3f" % (model_name, roc_auc_score(y_dev, predictions))
        print(text)
        write_to_file.write(text + "\n")

    write_to_file.close()
    print(f"Results saved to {write_to_path}")
    return pipeline

def get_data(test_size: int=0)-> tuple[pd.DataFrame, pd.DataFrame, list]:
    ''''
    Read in the gzipped json file and split the data into training and development sets--default split is 80/20.

    :param test_size: size of the test set
    :return: training and development sets, and dates
    '''
    # input_file = gzip.open("/deac/csc/classes/csc373/data/assignment_5/steam_reviews.json.gz")
    file = os.path.join(os.sep, 'deac', 'csc', 'classes', 'csc373', 'data', 'assignment_5', 'steam_reviews.json.gz')
    input_file = gzip.open(file)
    dataset = []
    for idx, l in enumerate(input_file):
        d = eval(l)
        dataset.append(d)
        if test_size > 0 and test_size < idx: break
    input_file.close()

    dates = []
    for i in range(len(dataset)): 
        dates.append(int(dataset[i]['date'][:4]))
        if test_size > 0 and test_size < i: break

    # 80/20 split
    train_data = dataset[:int(len(dataset)*0.8)]
    dev_data = dataset[int(len(dataset)*0.8):]

    return pd.DataFrame(train_data), pd.DataFrame(dev_data), dates

def main()-> None:
    parser = argparse.ArgumentParser(description='Assignment 5')
    parser.add_argument('-test_size', type=int, required=False, default=0, help='Size of the test set (optional)')
    args = parser.parse_args()

    if args.test_size > 0:
        print(f"Testing size: {args.test_size}...")
    elif args.test_size < 0:
        print("Test size must be a non-negative integer!")
        exit(1)

    train_data, dev_data, dates = get_data(test_size=args.test_size)

    utils.data_understanding(data=train_data, output_path="train_data_data_desc", save=True)
    utils.data_understanding(data=dev_data, output_path="dev_data_data_desc", save=True)

    # note: for each section below, create a branch new pipeline function (as well as their respective helper functions)
    # ESTIMATION








    # CLASSIFICATION








    # RECOMMENDATION



    exit(0)

if __name__ == '__main__':
    main()