#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from report.evaluator import Evaluator
from sklearn.metrics import accuracy_score


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
    myStupidClassifier = StupidRecognizer(data.trainingSet,
                                          data.validationSet,
                                          data.testSet)
    
    # parameters
    learnRate = 0.005
    maxEpochs = 20
    #epochNumber = 30
    
    xEpochs = []
    yAccuracyPerceptron = []
    yAccuracyLogistic = []
    # loop for gathering data for graph plotting
    for epochNumber in xrange(1, maxEpochs+1):

        myPerceptronClassifier = Perceptron(data.trainingSet,
                                            data.validationSet,
                                            data.testSet,
                                            learningRate=learnRate,#0.005,
                                            epochs=epochNumber)
        # Uncomment this to run Logistic Neuron Layer
        myLRClassifier = LogisticRegression(data.trainingSet,
                                            data.validationSet,
                                            data.testSet,
                                            learningRate=learnRate,#0.005,
                                            epochs=epochNumber#30
                                            )
    
        # Train the classifiers
        print("=========================")
        print("Training..")
    
        print("\nStupid Classifier has been training..")
        myStupidClassifier.train()
        print("Done..")
    
        print("\nPerceptron has been training..")
        myPerceptronClassifier.train()
        print("Done..")
    
        print("\nLogistic Regression has been training..")
        myLRClassifier.train()
        print("Done..")
    
        # Do the recognizer
        # Explicitly specify the test set to be evaluated
        stupidPred = myStupidClassifier.evaluate()
        perceptronPred = myPerceptronClassifier.evaluate()
        lrPred = myLRClassifier.evaluate()
    
        # Report the result
        print("=========================")
        evaluator = Evaluator()
    
        print("Result of the stupid recognizer:")
        #evaluator.printComparison(data.testSet, stupidPred)
        evaluator.printAccuracy(data.testSet, stupidPred)
    
        print("\nResult of the Perceptron recognizer:")
        #evaluator.printComparison(data.testSet, perceptronPred)
        evaluator.printAccuracy(data.testSet, perceptronPred)
    
        print("\nResult of the Logistic Regression recognizer:")
        #evaluator.printComparison(data.testSet, lrPred)
        evaluator.printAccuracy(data.testSet, lrPred)
    
    
    
        # accumulate plotting data
        xEpochs.append(epochNumber)
        yAccuracyPerceptron.append(accuracy_score(data.testSet.label, perceptronPred)*100)
        yAccuracyLogistic.append(accuracy_score(data.testSet.label, lrPred)*100)
            
        # === end of for loop ===
        
        
        
    # plot the graph
    plt.plot(xEpochs, yAccuracyPerceptron, marker='o', label='Perceptron')
    plt.plot(xEpochs, yAccuracyLogistic, marker='o', color='r', label='Logistic Neuron')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy [%]')
    plt.title('Performance on different epochs\n(using: testSet | learningRate: ' + str(learnRate) + ')')
    #plt.legend()
    plt.legend(loc=4)
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
    

if __name__ == '__main__':
    main()
