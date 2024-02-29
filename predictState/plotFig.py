import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd

def plotResult():
    returns = np.loadtxt("result/beh/behaveAccuracy.csv", delimiter=',')

    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.show()

plotResult()

def plotCombinedResults():
    returns_train = np.loadtxt("result/beh/losses_train.csv", delimiter=',')
    returns_valid = np.loadtxt("result/beh/losses_valid.csv", delimiter=',')


    plt.plot(returns_train, label='losses_train', color='blue')
    plt.plot(returns_valid, label='losses_valid', color='red')

    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Loss")

    plt.legend()

    plt.show()


plotCombinedResults()