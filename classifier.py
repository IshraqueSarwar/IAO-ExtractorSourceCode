import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
# import os
import cv2
from tqdm import tqdm
import glob
import pickle
import random	
			

# Differentiator net
class NET_(nn.Module):
	def __init__(self,):
		super().__init__()
		self.AVG_SHAPE = (32, 20)

		self.conv1 = nn.Conv2d(1, 32, kernel_size = (3,3), padding = (1,1))
		self.conv2 = nn.Conv2d(32, 64, kernel_size = (3,3), padding = (1,1))
		self.conv3 = nn.Conv2d(64, 128, kernel_size = (3,3), padding = (1,1))
		# self.conv4 = nn.Conv2d(128, 256, kernel_size = (3,3), padding = (1,1))
		# self.conv5 = nn.Conv2d(256, 128, kernel_size = (3,3), padding = (1,1))
		# self.conv6 = nn.Conv2d(128, 64, kernel_size = (3,3), padding = (1,1))

		x = torch.randn(self.AVG_SHAPE[0], self.AVG_SHAPE[1]).view(-1, 1, self.AVG_SHAPE[0], self.AVG_SHAPE[1])
		self._to_linear = None
		self.convs(x)

		self.fc1 = nn.Linear(self._to_linear, 64)
		self.fc2 = nn.Linear(64, 128)
		self.fc3 = nn.Linear(128, 23)
		# self.fc4 = nn.Linear(512, 23)


	def convs(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), (3,3), padding = 1)
		x = F.max_pool2d(F.relu(self.conv2(x)), (3,3),padding = 1)
		x = F.max_pool2d(F.relu(self.conv3(x)), (3,3),padding = 1)
		# x = F.max_pool2d(F.relu(self.conv4(x)), (3,3),padding = 1)
		# x = F.max_pool2d(F.relu(self.conv5(x)), (3,3),padding = 1)

		if self._to_linear is None:
			self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
		return x

	def forward(self, x):
		x = self.convs(x)
		x = x.view(-1, self._to_linear)

		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		# x = F.relu(self.fc3(x))
		x = self.fc3(x)

		return F.log_softmax(x, dim = 1)



# Recognizer net
class NET(nn.Module):
	def __init__(self,):
		super().__init__()
		self.AVG_SHAPE = (73, 20)

		self.conv1 = nn.Conv2d(1, 32, kernel_size = (3,3), padding = (1,1))
		self.conv2 = nn.Conv2d(32, 64, kernel_size = (3,3), padding = (1,1))
		self.conv3 = nn.Conv2d(64, 128, kernel_size = (3,3), padding = (1,1))
		self.conv4 = nn.Conv2d(128, 256, kernel_size = (3,3), padding = (1,1))
		x = torch.randn(self.AVG_SHAPE[0], self.AVG_SHAPE[1]).view(-1, 1, self.AVG_SHAPE[0], self.AVG_SHAPE[1])
		self._to_linear = None
		self.convs(x)

		self.fc1 = nn.Linear(self._to_linear, 64)		
		self.fc2 = nn.Linear(64, 128)
		self.fc3 = nn.Linear(128, 2)


	def convs(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), (3,3), padding = 1)
		x = F.max_pool2d(F.relu(self.conv2(x)), (3,3),padding = 1)
		x = F.max_pool2d(F.relu(self.conv3(x)), (3,3),padding = 1)
		x = F.max_pool2d(F.relu(self.conv4(x)), (3,3),padding = 1)

		if self._to_linear is None:
			self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
		return x

	def forward(self, x):
		x = self.convs(x)
		x = x.view(-1, self._to_linear)

		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)

		return F.log_softmax(x, dim = 1)










# NOTE: Test
if __name__ == "__main__":
	# for earlier models the shape will be shape[0], shape[1]
	# for newer model the shape will be shape[1], shape[0]


	train = input("Do you want to Train the model?(y/n): ")
	# Training
	BATCH_SIZE = 100
	EPOCHS = 30
	PCT_TEST = 0.15
	LR = 0.001
	MOMENTUM = 0.9 # for SGD optimizer
	shape = (32, 20)


	


	

	# Loading training data and preparing them
	training_data = np.load("training_images/TrainingData_v2.npy", allow_pickle = True)
	print("Preparing X from training data")
	X = torch.Tensor([i[0] for i in tqdm(training_data)]).view(-1,shape[1], shape[0])
	X/=255.0
	print("Preparing y from training data")
	y = torch.Tensor([i[1] for i in tqdm(training_data)])



	test_size = int(len(X)*PCT_TEST)
	train_X = X[:-test_size]
	train_y = y[:-test_size]
	test_X = X[-test_size:]
	test_y = y[-test_size:]

	
	if train.lower()=='y':
		# net = NET_()

		with open("models/Differentiator_Neural_Net-v3.pickle", 'rb') as n:
			net = pickle.load(n)
		# NOTE: Adam- optimizer and MSELoss- loss function
		# optimizer = optim.Adam(net.parameters(), lr = LR)
		# loss_func = nn.MSELoss()

		# NOTE: SGD optimizer and CrossEntropyLoss- loss function
		optimizer = optim.SGD(net.parameters(), lr = LR, momentum = MOMENTUM)
		loss_func = nn.CrossEntropyLoss()
	
		print("Training...")
		for epoch in range(EPOCHS):
			print(f"EPOCH#{epoch+1}:")
			for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
				batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, shape[1], shape[0])
				batch_y = train_y[i:i+BATCH_SIZE]

				net.zero_grad()
				output = net(batch_X)
				loss = loss_func(output, batch_y)
				loss.backward()
				optimizer.step()


			print(f"Loss of EPOCH#{epoch}: {loss}")
			print("Saving the model...")
			with open(f"models/model(SGD):EPOCH#{epoch}_lessConv.pickle", 'wb') as f:
				pickle.dump(net, f)

	else:
		with open(f"models/Differentiator_Neural_Net-v4.pickle", 'rb') as f:
			net = pickle.load(f)

		correct = 0
		length_x = len(test_X)
		print("Testing the model...")
		with torch.no_grad():
			for i in tqdm(range(length_x)):
				
				out = net(test_X[i].view(-1, 1, shape[1], shape[0]))
				# print(torch.argmax(out), torch.argmax(test_y[i]))
				if torch.argmax(out) == torch.argmax(test_y[i]):
					correct+=1
				else:
					print(test_y[i])
					plt.imshow(test_X[i])
					plt.show()
				# break

		print(f"Correct: {correct} out of {length_x}")
		print(f"accuracy: {100*(correct/length_x)}")


