import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')
a=np.array(dataset)

X = dataset.iloc[:,[0,1,2,3,4,6,5,7]].values
Y = dataset.iloc[:,8].values
xPredict = np.array(([1, 55, 1, 0, 0, 1, 63, 19]), dtype=float)


class Neural_Network(object):
  def __init__(self):
    #parameters
    self.inputSize = 8
    self.outputSize = 1
    self.hiddenSize = 3

    #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) 
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) 

  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set  weights
    self.z2 = self.sigmoid(self.z) # activation function
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = self.sigmoid(self.z3) # final activation function
    return o

  def sigmoid(self, s):
    # activation function
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
     #derivitive of igmoid
     return s* (1-s)

  def backward(self, X, Y, o):
        # backward propagate through the network
        self.o_error = Y - o # error in output
        self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

        self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
        self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights 
  def train (self, X, Y):
        o = self.forward(X)
        self.backward(X, Y, o)
  def saveWeights(self):
    np.savetxt("w1.txt", self.W1, fmt="%s")
    np.savetxt("w2.txt", self.W2, fmt="%s")
  def predict(self):
    print("Predicted data based on trained weights: ")
    print("Input (scaled): \n" + str(xPredict))
    print("Output: \n" + str(self.forward(xPredict)))
  def predict_new(self, xPredict):
    print("Predicted data based on trained weights: ")
    print("Input (scaled): \n" + str(xPredict))
    print("Output: \n" + str(self.forward(xPredict)))

NN = Neural_Network()
NN.saveWeights()
prediction = NN.predict()

for index, row in test_dataset.iterrows():
  if(index<10):
  
      data = pd.DataFrame( {'gender':test_dataset['gender'], 'age':test_dataset['age'],  'hypertension':test_dataset['hypertension'],  'heart_disease':test_dataset['heart_disease'],     
     'ever_married':test_dataset['ever_married'],  'Residence_type':test_dataset['Residence_type'],   'avg_glucose_level':test_dataset['avg_glucose_level'], 'bmi':test_dataset['bmi']} )
      
      test_dataset['predicted_stroke'] = str(NN.predict_new(data))
      test_dataset.to_csv('test_dataset.csv', index=False)
  else:
    exit()


