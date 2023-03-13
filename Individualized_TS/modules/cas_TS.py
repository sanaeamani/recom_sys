#!/usr/bin/env python
# coding: utf-8
#
#This file implements algorithm Cas TS. self.train trains the parameters on an input file df that includes data of users recommendations and their selected item and saves the new trained parameters into parameters_loc. self.predict returns the recommendations for a user with user_id passed as an input. The class uses user-ids to account for individualized diversification.
import csv
import numpy as np
import pickle
from scipy.stats import bernoulli, binom
import collections
import random
#
#
#
class RecSys(object):
    def __init__(self, items, K, L, alpha):
        self.K = K #number of recommendations per user
        self.alpha = alpha #diversity parameter
        self.day = 1 #current round of learning
        self.L = L #total number of mindfulness items, i.e., the size of the arm set
        self.num_users = 10000 #maximum number of user-ids
        self.repeat = {b: 0 for b in range(self.num_users)}
        self.regret = 0
        self.number_of_recomms = {b: np.zeros([2,self.L],dtype=int) for b in range(self.num_users)} #the first row is the number of times each item is being recommended in the current round of learning, and the second row is the number of times each item was recommended in the previous round of learning
        self.T1 = {b: np.ones([self.L]) for b in range(self.num_users)}
        self.T2 = {b: np.ones([self.L]) for b in range(self.num_users)}
        self.items = items
#
#
#
    def get_reward(self, a, weightt):
        return bernoulli(weightt[a]).rvs(1)
#
#
#
    def find_indices(self, list_to_check, item_to_find):
        indices = []
        for idx, value in enumerate(list_to_check):
            if value == item_to_find:
                indices.append(idx)
        return indices
#
#
#
    def my_sort(self, a, K):
        b = collections.Counter(a)
        ind = a.argsort()[-K:][::-1]
        for i in range(len(ind)):
            if b[a[ind[i]]]>1:
                indd = self.find_indices(a.tolist(), a[ind[i]])
                ind[i] = random.choice(indd)
        return ind
#
#
#
    def train(self, parameters_loc, df, from_scratch):
        if from_scratch or self.day==1:
            w_hat = np.zeros([self.L])
            T = np.ones([self.L])
        else:
            with open(parameters_loc, 'rb') as f:
                [w_hat, T] = pickle.load(f)
        if self.day>1:
            sample_size = df.shape[0]
            for sample in range(sample_size):
                for i in range(1,self.K+1):
                    self.number_of_recomms[df['user_id'][sample]][0][self.find_indices(self.items['IDENTIFIER'], df['choice{}'.format(i)][sample])] +=1
                selected_indx = self.find_indices(self.items['IDENTIFIER'], df['selected'][sample])
                w_hat[selected_indx] = (self.T1[df['user_id'][sample]][selected_indx]+T[selected_indx]*w_hat[selected_indx])/(T[selected_indx]+1)
                T[selected_indx]+=1
                i = 1
                while(df['choice{}'.format(i)][sample]!=df['selected'][sample]):
                    w_hat[self.find_indices(self.items['IDENTIFIER'], df['choice{}'.format(i)][sample])] = (T[self.find_indices(self.items['IDENTIFIER'], df['choice{}'.format(i)][sample])]*w_hat[self.find_indices(self.items['IDENTIFIER'], df['choice{}'.format(i)][sample])])/(T[self.find_indices(self.items['IDENTIFIER'], df['choice{}'.format(i)][sample])]+1)
                    T[self.find_indices(self.items['IDENTIFIER'], df['choice{}'.format(i)][sample])]+=1
                    i+=1
                self.T1[df['user_id'][sample]] = self.T2[df['user_id'][sample]]
                for i in range(self.L):
                    self.T2[df['user_id'][sample]][i] = (1+self.number_of_recomms[df['user_id'][sample]][0,i])**(self.alpha)
                self.repeat[df['user_id'][sample]] = 0
                for i in range(self.L):
                    self.repeat[df['user_id'][sample]] += min(self.number_of_recomms[df['user_id'][sample]][0][i], self.number_of_recomms[df['user_id'][sample]][1][i])
                self.number_of_recomms[df['user_id'][sample]][1] = self.number_of_recomms[df['user_id'][sample]][0]
                self.number_of_recomms[df['user_id'][sample]][0] = np.zeros([self.L])
        with open(parameters_loc, 'wb') as f:
            pickle.dump([w_hat, T], f)
        self.day+=1
#
#
#
    def predict(self, parameters_loc, user_id):
        with open(parameters_loc, 'rb') as f:
            [w_hat, T] = pickle.load(f)
        recommendations = np.zeros([self.K],dtype=int)
        Z = np.random.normal(0, 1)
        U = (w_hat/self.T2[user_id])+Z*(np.log(self.day*1000)/T)/self.T2[user_id]
        recommendations = self.my_sort(U, self.K)
        return self.items['IDENTIFIER'][recommendations].tolist()
#
#
#
    def make_recommendation_and_save_new_data(self, weight, recommendations, data_loc, user_id):
        recs = np.zeros([self.K],dtype=int)
        for i in range(self.K):
            recs[i] = self.find_indices(self.items['IDENTIFIER'], recommendations[i])[0]
        weightt = weight/self.T2[user_id]
        indd = weightt.argsort()[-self.K:][::-1]
        optimal_reward = 1-np.prod(1-weightt[indd])
        self.regret = optimal_reward-1+np.prod(1-weightt[recs])
        clicks = []
        sample_data = []
        for i in range(self.K):
            clicks.append(int(self.get_reward(recs[i], weightt)))
            sample_data.append(self.items['IDENTIFIER'][recs[i]])
        if sum(clicks)>0:
            indx = self.find_indices(clicks, 1)[0]
            sample_data = [self.items['IDENTIFIER'][recs[indx]]]+sample_data
            with open(data_loc, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(sample_data+[user_id])
#
#
#