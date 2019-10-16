from __future__ import absolute_import, division, print_function

import numpy as np
from tensorflow.python.keras.callbacks import Callback
from collections import OrderedDict
from sklearn.metrics import auc, log_loss, precision_recall_curve, roc_auc_score, average_precision_score



ltrdict = {'a':[1,0,0,0],
           'c':[0,1,0,0],
           'g':[0,0,1,0],
           't':[0,0,0,1],
           'n':[0,0,0,0],
           'A':[1,0,0,0],
           'C':[0,1,0,0],
           'G':[0,0,1,0],
           'T':[0,0,0,1],
           'N':[0,0,0,0]}

def one_hot_encode(seqs):
    encoded_seqs=np.array([[ltrdict.get(x,[0,0,0,0]) for x in seq] for seq in seqs])
    encoded_seqs=np.expand_dims(encoded_seqs,1)
    return encoded_seqs

def randomize(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b

def loss(labels, predictions):
    return log_loss(labels, predictions)


def positive_accuracy(labels, predictions, threshold=0.5):
    return 100 * (predictions[labels] > threshold).mean()


def negative_accuracy(labels, predictions, threshold=0.5):
    return 100 * (predictions[~labels] < threshold).mean()


def balanced_accuracy(labels, predictions, threshold=0.5):
    return (positive_accuracy(labels, predictions, threshold) +
            negative_accuracy(labels, predictions, threshold)) / 2


def auROC(labels, predictions):
    return roc_auc_score(labels, predictions)


def auPRC_careful(labels, predictions):
    auc_careful = average_precision_score(labels, predictions)
    return auc_careful

def auPRC_trapezoid(labels, predictions):
    precision_trapezoid,recall_trapezoid=precision_recall_curve(labels,predictions)[:2]
    auc_trapezoid=auc(recall_trapezoid,precision_trapezoid) 
    return auc_trapezoid



def recall_at_precision_threshold(labels, predictions,precision_threshold):
    precision, recall = precision_recall_curve(labels, predictions)[:2]
    return 100 * recall[np.searchsorted(precision - precision_threshold, 0)]


class ClassificationResult(object):

    def __init__(self, labels, predictions, task_names=None):
        assert labels.dtype == bool
        self.results = [OrderedDict((
            ('Loss', loss(task_labels, task_predictions)),
            ('Balanced accuracy', balanced_accuracy(
                task_labels, task_predictions)),
            ('auROC', auROC(task_labels, task_predictions)),
            ('auPRC Careful', auPRC_careful(task_labels, task_predictions)),
            ('auPRC Trapezoid', auPRC_trapezoid(task_labels,task_predictions)),
            ('Recall at 5% FDR', recall_at_precision_threshold(
                task_labels, task_predictions, 0.95)),
            ('Recall at 10% FDR', recall_at_precision_threshold(
                task_labels, task_predictions, 0.9)),
            ('Recall at 20% FDR', recall_at_precision_threshold(
                task_labels, task_predictions, 0.8)),
            ('Num Positives', task_labels.sum()),
            ('Num Negatives', (1 - task_labels).sum())
        )) for task_labels, task_predictions in zip(labels.T, predictions.T)]
        self.task_names = task_names
        self.multitask = labels.shape[1] > 1

    def __str__(self):
        return '\n'.join(
            '{}Loss: {:.4f}\tBalanced Accuracy: {:.2f}%\t '
            'auROC: {:.3f}\t auPRC Careful: {:.3f}\t auPRC Trapezoidal: {:.3f}\n\t'
            'Recall at 5%|10%|20% FDR: {:.1f}%|{:.1f}%|{:.1f}%\t '
            'Num Positives: {}\t Num Negatives: {}'.format(
                '{}: '.format('Task {}'.format(
                    self.task_names[task_index]
                    if self.task_names is not None else task_index))
                if self.multitask else '', *results.values())
            for task_index, results in enumerate(self.results))

    def __getitem__(self, item):
        return np.array([task_results[item] for task_results in self.results])


class MetricsCallback(Callback):
    def __init__(self, train_data, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.train_data = train_data    
    def on_epoch_end(self, epoch, logs={}):            
        X_train = self.train_data[0]
        y_train = self.train_data[1]
        
        X_val = self.validation_data[0]
        y_val = self.validation_data[1]

        y_train_pred=self.model.predict(X_train)
        y_val_pred=self.model.predict(X_val)

        train_classification_result=ClassificationResult(y_train,y_train_pred)
        val_classification_result=ClassificationResult(y_val,y_val_pred)
        print("Training Data:") 
        print(train_classification_result)
        print("Validation Data:") 
        print(val_classification_result)


