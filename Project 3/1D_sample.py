import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import os
import time
import pylab as pl
from roipoly import roipoly
from PIL import Image
import glob
from os import listdir
from PIL import Image as PImage
import imageio

def pdf(data, mean: float, variance: float):
  # A normal continuous random variable.
  s1 = 1/(np.sqrt(2*np.pi*variance))
  s2 = np.exp(-(np.square(data - mean)/(2*variance)))
  # log_likelihood_pdf = (-(1/2) * np.log(2*np.pi)) -((1/2)*np.log(variance**2))-((1/(2*(variance**2))) * (np.square(data-mean)))
  g  = s1*s2
  print("g=",g)
  print(g.shape)
  print('-----')
  return s1*s2
  # return log_likelihood_pdf
# define the number of clusters to be learned


def gaussian():
    #sample mu and sigma values
    mu1 = 0
    mu2 = 3
    mu3 = 6
    sigma1 = 2
    sigma2 = 0.5
    sigma3 = 3

    s1 = np.random.normal(mu1, sigma1, 100)
    s2 = np.random.normal(mu2, sigma2, 100)
    s3 = np.random.normal(mu3, sigma3, 100)
    X = np.array(list(s1) + list(s2) + list(s3))

    np.random.shuffle(X)
    #print("after shuffle X=",X)
    bins = np.linspace(np.min(X), np.max(X), 100)

    plt.figure(figsize=(10, 7))
    plt.xlabel("$x$")
    plt.ylabel("pdf")
    plt.scatter(X, [0.005] * len(X), color='navy', s=30, marker=2, label="Train data")

    plt.title('Combined PDF and individual pdf ')
    # combined pdf func
    plt.plot(bins, pdf(bins, mu1, sigma1) + pdf(bins, mu2, sigma2) + pdf(bins, mu3, sigma3)  , color='red', lw = 10,label="True pdf")

    # individual pdf funs
    plt.plot(bins, pdf(bins, mu1, sigma1), color='green', label="True pdf")
    plt.plot(bins, pdf(bins, mu2, sigma2), color='blue')
    plt.plot(bins, pdf(bins, mu3, sigma3), color='black')

    # print("mean=",means)
    # plt.legend()
    # plt.plot()
    return bins,mu1,mu2,mu3,sigma1,sigma2,sigma3,X

def em():
    #s = np.random.normal(mu, sigma, 1000)
    #X = data_points()
    bins, mu1, mu2, mu3, sigma1, sigma2, sigma3, X = gaussian()
    k = 3
    weights = np.ones((k)) / k
    means = np.random.choice(X, k)
    variances = np.random.random_sample(size=k)
    # visualize the training data
    k=3
    z=0

    eps = 0.00021

    iterations = 100
    iterations_completed = 0

    while iterations_completed<iterations:
        z=z+1
        if z==iterations-1:

            plt.figure(figsize=(10, 6))
            axes = plt.gca()
            plt.xlabel("$x$")
            plt.ylabel("pdf")
            plt.title("Iteration {}".format(iterations_completed))
            plt.scatter(X, [0.005] * len(X), color='navy', s=30, marker=2, label="Train data")

            # combined pdf func
            plt.plot(bins,pdf(bins, means[0], variances[0]) + pdf(bins, means[1], variances[1]) + pdf(bins, means[2], variances[2]) , color='green', lw=8, label="True pdf")
            plt.plot(bins, pdf(bins, mu1, sigma1) + pdf(bins, mu2, sigma2) + pdf(bins, mu3, sigma3), color='red', lw=8,
                     label="True pdf")

            # individual funcs
            plt.plot(bins, pdf(bins, means[0], variances[0]), color='blue', label="Cluster 1")
            plt.plot(bins, pdf(bins, means[1], variances[1]), color='pink', label="Cluster 2")
            plt.plot(bins, pdf(bins, means[2], variances[2]), color='magenta', label="Cluster 3")

            plt.legend(loc='upper left')
            plt.show()
            print("mean=", means)

            #save figure plot

            # plt.savefig("img_{0:02d}".format(step), bbox_inches='tight')
        # if(z==100):
        #     plt.show()
    # calculate the maximum likelihood of each observation xi
        likelihood_list = []
        log_likelihood = []

        # Expectation step
        for j in range(k):
            likelihood_list.append(pdf(X, means[j], np.sqrt(variances[j])))
            likelihood_value = np.sum(np.log(likelihood_list))
            log_likelihood.append(likelihood_value)
        likelihood_list = np.array(likelihood_list)

        b = []

        # Maximization step
        for j in range(k):
            # use the current values for the parameters to evaluate the posterior
            # probabilities of the data to have been generanted by each gaussian
            b.append((likelihood_list[j] * weights[j]) / (np.sum([likelihood_list[i] * weights[i] for i in range(k)], axis=0) + eps))
            # b.append((np.log(likelihood[j]) + np.log(weights[j])))  # / (np.sum([likelihood[i] * weights[i] for i in range(k)], axis=0) + eps))

            # updage mean and variance
            means[j] = np.sum(b[j] * X) / (np.sum(b[j] + eps))
            variances[j] = np.sum(b[j] * np.square(X - means[j])) / (np.sum(b[j] + eps))

            # update the weights
            weights[j] = np.mean(b[j])

        if np.abs(likelihood_value - log_likelihood[-2]) < eps:
            print('here')
            break
        iterations_completed+=1
em()