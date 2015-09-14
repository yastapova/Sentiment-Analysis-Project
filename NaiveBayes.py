#!/usr/bin/env python

import pandas as pd
import nltk
import math
import re
import numpy as np
from nltk.corpus import stopwords

class NaiveBayes:
	
	def __init__(self, training, sentimentValues, categories, vocab, test, testSent):
		data = self.separateTestTrain(training, sentimentValues)
		self._categorized = self.separateData(data[2], data[3], categories) #The words
		self._categories = categories #the sentiment value
		self._vocab = vocab 
		self._addOn = 0
		self.laplaceSmoothing(data[2], 10, data[0], data[1])

	###########################################################################################	
	# Separates the training data. creates a dictionary with the key being the sentiment value#
	# and the value being a list of all the sentences that have that sentiment value		  #
	###########################################################################################	
	def separateData(self, training, senValues, categories):
		categorized = dict()
		for i in range(0, len(categories)):
			for j in range(0, len(training)):
				if senValues[j] == i:
					categorized.setdefault(i, []).append(training[j])
		return categorized
		
	
	###########################################################################################	
	# Calculates the prior probabilities ( P(Y) ) The probability that any given sentence 	  #
	# is part of any category																  #
	###########################################################################################	
	def calculatePriors(self):
		prior = []
		for i in range(0, len(self._categories)):
			prior.append((len(self._categorized[i]) + self._addOn) / float(self._numSentences))
		return prior
	
	###########################################################################################	
	# Used for testing. Prints some info about the data										  #
	###########################################################################################	
	def getCurrentState(self):
		
		print("Num Sentences: ")
		print(self._numSentences)
		print("Add on value: %i" % self._addOn)
		for i in range(0, len(self._categories)):
			print("Num Category %i" % i)
			print("\tNum Entries %i" % (len(self._categorized[i]) + self._addOn))
			print("\tPrior prob %f" % self._prior[i])

	###########################################################################################	
	#Should separate the data into test and train data. Not used may not work properly		  #
	###########################################################################################	
	def separateTestTrain(self, training, sentiment):
		test = []
		testSent = []
		train = []
		trainSent = []
		i = 0;
		for j in training:
			if (i % 30) == 0:
				test.append(j)
				testSent.append(sentiment[i])
			else: 
				train.append(j)
				trainSent.append(sentiment[i])
			i += 1
		return [test, testSent, train, trainSent]
			
	###########################################################################################	
	# Gets the P(Wi|Y) for all the words in the vocabulary and for each of the categories	  #
	#The probability that given a certain word that it is in that category					  #
	###########################################################################################	
	def calculateWordProb(self, addon):
		temp = float(addon)
		count = []
		prob = dict()
		#for all words in the vocabulary
		for i in range(0, len(self._vocab)): 
		
			#for every category (0 - 4)
			for j in range(0 , len(self._categorized)): 
				temp = self._addOn	
				
				#goes thru each sentence in the category
				#and counts the occurrences of the word
				for k in range(0, len(self._categorized[j])):				
					if(self._categorized[j][k].find(self._vocab[i]) != -1):	
						temp += 1						#if the word is found increment the count
				#End innermost for loop (For every sentence in the category)

				#Appends the probability of a certain word occurring in a category
				count.append(float(temp) / float(len(self._categorized[j]) + self._addOn))
				
			# end middle loop (for each category)	
				
			#Add the probability of each word given the category
			#For example Given the word "test" it showed up in 12% of sentences labeled 
			# as 0, 8% of sentences labeled as 1 etc.
			# prob["test"] = [ 12, 8, 3, 9, 10]
			prob[self._vocab[i]] = count
			count = []
			# end Outer loop (for each word in the vocab)
			
		
		return prob
		
	###########################################################################################	
	#Takes a sentence and will return a sentiment value										  #
	###########################################################################################	
	def classifySentence(self, sentence):
		#set the priors
		score = [math.log(self._prior[0]), math.log(self._prior[1]), math.log(self._prior[2]), math.log(self._prior[3]), math.log(self._prior[4])]
		
		#split sentence into the words
		sent = sentence.split()
		
		prob = self._probs
		
		#Goes thru each word in the sentence and adds up the logs of the probability that
		#the given word is in each category
		for i in sent:
				if prob.get(i) == None:
					continue
				if prob[i][0] != 0:
					score[0] += math.log(prob[i][0])
				if prob[i][1] != 0:
					score[1] += math.log(prob[i][1])
				if prob[i][2] != 0:
					score[2] += math.log(prob[i][2])
				if prob[i][3] != 0:
					score[3] += math.log(prob[i][3])
				if prob[i][4] != 0:
					score[4] += math.log(prob[i][4])

		#returns the score based on which index(sentimentValue) has the highest probability
		return score.index(max(score))
		
	###########################################################################################	
	#for all the values 0 - k creates a classifier with that added fake entries				  #
	###########################################################################################	
	def laplaceSmoothing(self, training, k, test, testSen):
		accuracy = []
		for i in range(0, k):
			self._addOn = i
			self._numCat = len(self._categories)
			self._numSentences = len(training) + (self._numCat * self._addOn) #adding the fake data points
			self._prior = self.calculatePriors() 
			self._probs = self.calculateWordProb(self._addOn)
			self.getCurrentState()
			accuracy.append(self.testClassifier(test, testSen))
		best = accuracy.index(max(accuracy))
		print (accuracy)
		print("Best K value %i" % (best))
		print("percent correct %f" % max(accuracy))
		return self._probs
		
		
	###########################################################################################	
	#Takes the test data to test the accuracy of the classifier. compares the classifier score#
	# with the actual score and will print the accuracy										  #
	###########################################################################################	
	def testClassifier(self, test, sentiment):
		j = 0
		print("TESTING")
		count = 0
		for i in test:
			classifierScore = self.classifySentence(i)			
			actualScore = sentiment[j]
			if classifierScore == actualScore:
				count += 1
			j+= 1
		print("Precent Correct %i" % ((float(count) / len(test)) * 100))
		return ((float(count) / len(test)) * 100)
		


