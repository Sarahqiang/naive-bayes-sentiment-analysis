import inline as inline
import matplotlib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from dataclean import sentiment
from dataclean import traindata
from dataclean import testsentiment
from dataclean import testdata
import numpy as np
from math import e
import itertools

import matplotlib.pyplot as plt

from sklearn import datasets


class GaussianNaiveBayes:

    def __init__(self):
        return

    def fit(self, x, y,alpha):
        N, D = x.shape
        C = np.max(y) + 1

        mu, sigma = np.zeros((C, D)), np.zeros((C, D))
        Nc = np.zeros(C)

        for c in range(C):
            x_c = x[y == c]
            Nc[c] = x_c.shape[0]
            mu[c, :] = np.mean(x_c, 0)
            sigma[c, :] = alpha + np.std(x_c, 0)

        self.mu = mu
        self.sigma = sigma
        self.pi = (Nc + 1) / (
                    N + C)
        return self
def logsumexp(Z):                                                # dimension C x N
    Zmax = np.max(Z,axis=0)[None,:]                              # max over C
    log_sum_exp = Zmax + np.log(np.sum(np.exp(Z - Zmax), axis=0))
    return log_sum_exp

def predict(self, xt):
    Nt, D = xt.shape
    log_prior = np.log(self.pi)[:, None]
    log_likelihood = -.5 * np.log(2*np.pi) - np.log(self.sigma[:,None,:]) -.5 * (((xt[None,:,:] - self.mu[:,None,:])/self.sigma[:,None,:])**2)
    # now we sum over the feature dimension to get a C x N matrix (this has the log-likelihood for each class-test point combination)
    log_likelihood = np.sum(log_likelihood, axis=2)
    # posterior calculation
    log_posterior = log_prior + log_likelihood
    posterior = np.exp(log_posterior - logsumexp(log_posterior))
    return posterior.T                                                  # dimension N x C

GaussianNaiveBayes.predict = predict

np.random.seed(1234)
categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med','sci.space','sci.crypt','rec.autos','sci.electronics']
twenty_train = fetch_20newsgroups(subset='train',  categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)


count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_new_counts = count_vect.transform(twenty_test.data)

count_vect.vocabulary_.get(u'algorithm')
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)





# model = GaussianNaiveBayes()
# model.fit(X_train_tfidf.toarray(),twenty_train.target,1000)
# y_prob = model.predict(X_new_tfidf.toarray())
# y_pred = np.argmax(y_prob, 1)
# accuracy = np.sum(y_pred == twenty_test.target)/y_pred.shape[0]
# print(f'test accuracy: {accuracy}')

model = GaussianNaiveBayes()
model.fit(traindata,sentiment,1)
y_prob = model.predict(testdata)
y_pred = np.argmax(y_prob, 1)
accuracy = np.sum(y_pred == testsentiment)/y_pred.shape[0]
print(f'test accuracy: {accuracy}')

