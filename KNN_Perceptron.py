import pandas as pd
import numpy as np
import math
import time

#Data with features and target values
dataset = pd.read_csv("data.csv")

#Given Helper Functions as part of the Assignment
#========================================== Data Helper Functions ==========================================

#Normalize values between 0 and 1
#dataset: Pandas dataframe
#categories: list of columns to normalize, e.g. ["column A", "column C"]
#Return: full dataset with normalized values
def normalizeData(dataset, categories):
    normData = dataset.copy()
    col = dataset[categories]
    col_norm = (col - col.min()) / (col.max() - col.min())
    normData[categories] = col_norm
    return normData

#Encode categorical values as mutliple columns (One Hot Encoding)
#dataset: Pandas dataframe
#categories: list of columns to encode, e.g. ["column A", "column C"]
#Return: full dataset with categorical columns replaced with 1 column per category
def encodeData(dataset, categories):
	return pd.get_dummies(dataset, columns=categories)

#Split data between training and testing data
#dataset: Pandas dataframe
#ratio: number [0, 1] that determines percentage of data used for training
#Return: (Training Data, Testing Data)
def trainingTestData(dataset, ratio):
	tr = int(len(dataset)*ratio)
	return dataset[:tr], dataset[tr:]

#Convenience function to extract Numpy data from dataset
#dataset: Pandas dataframe
#Return: features numpy array and corresponding labels as numpy array
def getNumpy(dataset):
	features = dataset.drop(["can_id", "can_nam","winner"], axis=1).values
	labels = dataset["winner"].astype(int).values
	return features, labels

#Convenience function to extract data from dataset (if you prefer not to use Numpy)
#dataset: Pandas dataframe
#Return: features list and corresponding labels as a list
def getPythonList(dataset):
	f, l = getNumpy(dataset)
	return f.tolist(), l.tolist()

#Calculates accuracy of your models output.
#solutions: model predictions as a list or numpy array
#real: model labels as a list or numpy array
#Return: number between 0 and 1 representing your model's accuracy
def evaluate(solutions, real):
	predictions = np.array(solutions)
	labels = np.array(real)
	return (predictions == labels).sum() / float(labels.size)

#===========================================================================================================

class KNN:
	def __init__(self):
		#initialize empty variables to hold training data
		self.model_features = None
		self.model_labels = None
		return

	def train(self, features, labels):
		#input is list/array of features and labels

		#store training data
		self.model_features = features
		self.model_labels = labels
		return

	def predict(self, features):
		#set k value
		k = 5

		prediction_lst = []

		#iterate through features of unseen data
		for item in features:

			distance_lst = [(np.inf, None)] * k

			#iterate through training model features
			for i in range(len(self.model_features)):

				#calcuate distance and store in distance array with label
				distance = Euclidean_distance(item, self.model_features[i])
				if distance <= distance_lst[len(distance_lst) - 1][0]:
					distance_lst.append((distance, self.model_labels[i]))

					# sort distances
					distance_lst.sort(key=lambda x: x[0])

					# slice distance list to keep to k
					distance_lst = distance_lst[:k]

			#tally labels for k nearest neighbors
			Win = 0
			Lose = 0
			for item in distance_lst:
				if item[1] == 1:
					Win += 1
				else:
					Lose += 1

			# append predicted target to prediction_lst
			if Win > Lose:
				prediction_lst.append(1)
			else:
				prediction_lst.append(0)

		#Return list/array of predictions where there is one prediction for each set of features
		return prediction_lst

def Euclidean_distance(test_item, model_item):

	dist_sum = 0
	for i in range(len(test_item)):
		dist_sum += abs(test_item[i] - model_item[i]) ** 2

	dist = math.sqrt(dist_sum)
	return dist

def preprocessKNNPerceptronMLP(dataset):
	#One Hot Encoding for categorical data
	features_list_to_encode = ["can_off", "can_inc_cha_ope_sea"]
	encoded_dataset = encodeData(dataset, features_list_to_encode)

	#normalize continuous data
	features_list_to_normalize = ["net_ope_exp", "net_con", "tot_loa"]
	normalized_encoded_dataset = normalizeData(encoded_dataset, features_list_to_normalize)

	#generate numpy arrays
	features, labels = getNumpy(normalized_encoded_dataset)

	#return numpy arrays of features and labels
	return features, labels

###test code

kNN = KNN()

#divide dataset into training and test data
train_dataset, test_dataset = trainingTestData(dataset, .80)

#preprocess train data
train_features, train_labels = preprocessKNNPerceptronMLP(train_dataset)

#train KNN model
kNN.train(train_features, train_labels)

#preprocess test data
test_features, test_labels = preprocessKNNPerceptronMLP(test_dataset)

#predict labels
predictions = kNN.predict(test_features)

#evaluate predictions
accuracy = evaluate(predictions, test_labels)
print(accuracy)

class Perceptron:
	def __init__(self):
		self.weight_vector = []
		self.bias_weight = 0
		self.bias = 1
		return

	def train(self, features, labels):
		# input is list/array of features and labels
		learning_rate = .01

		#randomly initialize weight vector
		for i in range(len(features[0])):
			self.weight_vector.append(np.random.uniform(low=0.0, high=0.25))

		#randomly initialize bias weight
		self.bias_weight = np.random.uniform(low=0.0, high=0.25)

		start_time = time.time()
		#change time limit later
		time_limit = 45
		makingErrors = True
		while makingErrors == True and (time.time() - start_time) < time_limit: #while make errors on training set
			makingErrors = False
			for row in range(len(features)):
				#calculate dot product
				dot_sum = np.dot(self.weight_vector, features[row])

				#add bias
				dot_sum += self.bias * self.bias_weight

				# pass through sigmoid function
				sigmoid_result = sigmoid(dot_sum)

				#determine class
				if sigmoid_result > 0.5:
					predict = 1 #win class is 1
				else: #lose class is 0
					predict = 0 #Lose

				#check prediction against label
				if predict != labels[row]: #wrong, update weights
					makingErrors = True
					#get correct classification value
					if labels[row] == 1:
						correct_class = 1
					else:
						correct_class = -1
					#update weight vector and bias weight
					for val in range(len(self.weight_vector)):
						self.weight_vector[val] = self.weight_vector[val] + learning_rate * correct_class * features[row][val]
						self.bias_weight = self.bias_weight + learning_rate * correct_class * self.bias
		return

	def predict(self, features):

		prediction_lst = []

		for row in features:

			# calculate dot product
			sum_dot = np.dot(self.weight_vector, row)

			#add bias
			sum_dot += self.bias * self.bias_weight

			#pass through sigmoid function
			sigmoid_result = sigmoid(sum_dot)

			#check if meets activation threshhold
			if sigmoid_result > 0.5: #win class is 1
				prediction_lst.append(1)
			else: #lose class is 0
				prediction_lst.append(0)

		#Return list/array of predictions where there is one prediction for each set of features
		return prediction_lst

def sigmoid(x):

	return float(1) / (1 + np.exp(-x))

###test code

perceptron = Perceptron()

#divide dataset into training and test data
train_dataset, test_dataset = trainingTestData(dataset, .80)

#preprocess train data
train_features, train_labels = preprocessKNNPerceptronMLP(train_dataset)

#train model
perceptron.train(train_features, train_labels)

#preprocess test data
test_features, test_labels = preprocessKNNPerceptronMLP(test_dataset)

#predict labels
predictions = perceptron.predict(test_features)

#evaluate predictions
accuracy = evaluate(predictions, test_labels)
print(accuracy)
