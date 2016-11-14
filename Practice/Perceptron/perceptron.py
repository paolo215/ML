import pandas as pd
import numpy as np


import string
import random

letters = {}
testing = {}
training = {}
results = {}


class Perceptron(object):
	def __init__(self, bias=1, learningRate=0.2):
		self.bias = bias
		self.learningRate = learningRate
		self.weights = []
	


	

	def train(self, pos, neg):
		
		self.weights = [random.random() * 2 - 1 for _ in range(len(pos[0]))]	
		
		currentAccuracy = 0
		previousAccuracy = 0
		deltaAccuracy = 1000
		countIter = 0
		while deltaAccuracy > 0.01:
			previousAccuracy = currentAccuracy
			currentAccuracy = self.epoch(list(pos),list(neg))

			deltaAccuracy = abs(currentAccuracy - previousAccuracy)


	def epoch(self, positives, negatives):
		correct = 0
		total = 0

		while len(positives) > 0 or len(negatives) > 0:
			inst = None
			actual = None
			if positives and random.random() > 0.5:
				inst = positives.pop()
				actual = 1
			elif negatives:
				inst = negatives.pop()
				actual = -1
			else:
				break				
			result = self.sign(self.calculateXTimesW(inst, self.weights))

			if result == actual:
				correct += 1
			else:
				self.adjustWeights(inst, result, actual)
			total += 1
		

		if total == 0:
			return 0
		return float(correct) / total

	
	def calculateXTimesW(self, x, w):
		return sum([x[i] * w[i] for i in range(len(x))])


	def sign(self, value):
		if value < 0:
			return -1
		return 1

	def adjustWeights(self, inst, result, actual):
		diff = actual - result
		for i in range(len(self.weights)):
			delta = self.learningRate * diff * inst[i]
			self.weights[i] += delta



	def test(self, pos, neg, labels):
		confusion = {}
		confusion["truePos"] = 0
		confusion["falsePos"] = 0
		confusion["trueNeg"] = 0
		confusion["falseNeg"] = 0
		total = 0

		for inst in pos:
			result = self.sign(self.calculateXTimesW(inst, self.weights))
			if result == 1:
				confusion["truePos"] += 1
			else:
				confusion["falseNeg"] += 1
			total += 1

		for inst in neg:
			result = self.sign(self.calculateXTimesW(inst, self.weights))
			if result == -1:
				confusion["trueNeg"] += 1
			else:
				confusion["falsePos"] += 1
			total += 1

		accuracy = (float( confusion["truePos"]) + confusion["trueNeg"] ) / total

		if accuracy > 0.50:
			return labels[0], accuracy
		elif accuracy < 0.50:
			return labels[1], accuracy
		else:
			return labels[int(random.random()) % 1], accuracy


def setup(dataFileStr, bias=1):
	global letters
	global training
	global testing
	letters = {}
	training = {}
	testing = {}
	dataFile = open(dataFileStr, "r")

	data = dataFile.read().replace("\r", "").split("\n")
	dataFile.close()

	
	for line in data:
		if not line: continue
		inst = line.split(",")
		

		letter = inst[0]
		if letter not in letters:
			letters[letter] = []
		newInst = map(float, inst[1:])
		newInst = [f / 15 for f in newInst + [bias]] 
		letters[letter].append(newInst)
		
	# split
	for letter in letters:
		random.shuffle(letters[letter])
		count = len(letters[letter])
	
		training[letter] = letters[letter][0:int(count/2)]
		testing[letter] = letters[letter][int(count/2):]


def main():
	setup("letter-recognition.data")
	asciiUpper = list(string.ascii_uppercase)
	
	firstTime = True
	accuracy = []
	for a in asciiUpper:
		row = []
		for b in asciiUpper:
			result = None
			if a == b: continue
			p = Perceptron()
			p.train(training[a], training[b])
			output = p.test(testing[a], training[b], (a,b))
			result = output[0]
			accuracy.append(output[1])
			row += result

	print sum(accuracy) / float(len(accuracy))


if __name__ == "__main__":
	main()
