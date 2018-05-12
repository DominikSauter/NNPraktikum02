# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier
from model.logistic_layer import LogisticLayer
from sklearn.metrics import accuracy_score

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    learningRate : float
    epochs : positive int
    """

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        
        
        self.logisticLayer = LogisticLayer(len(self.trainingSet.input[0]),
                                           1)
                                           #None,
                                           #'sigmoid',
                                           #False)
        
        #self.logisticLayer1 = LogisticLayer(len(self.trainingSet.input[0]),
        #                                   16)


    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        # Here you have to implement training method "epochs" times
        # Please using LogisticLayer class
        
        for epoch in xrange(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))

            self._train_one_epoch()

            if verbose:
                accuracy = accuracy_score(self.validationSet.label,
                                          self.evaluate(self.validationSet))
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy*100))
                print("-----------------------------")
        
        
    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """
        
        # for each training example do one forward pass (single layered neural network)
        for img, label in zip(self.trainingSet.input, self.trainingSet.label):
            
            #output1 = self.logisticLayer1.forward(img, False)
           
            output = self.logisticLayer.forward(img, False)
            
            # as this is the output layer: compute the output delta and pass it to layer
            delta = (label - output[0]) * output[0] * (1 - output[0])
            self.logisticLayer.computeDerivative([delta], None)
           
            #self.logisticLayer1.computeDerivative(self.logisticLayer.deltas, self.logisticLayer.weights)
            
            # online learning: updating weights after seeing 1 instance
            # if we want to do batch learning, accumulate the error
            # and update the weight outside the loop
            
            # update the weights of the layer
            self.logisticLayer.updateWeights(self.learningRate)
            
            #self.logisticLayer1.updateWeights(self.learningRate)
            
            
    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """

        # Here you have to implement classification method given an instance
        
        #output1 = self.logisticLayer1.forward(testInstance, True)
        
        output = self.logisticLayer.forward(testInstance, True)
        # output is in interval [0, 1], generally if it is > 0.5 it is counted as True
        return (output[0] > 0.5)
        

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))
