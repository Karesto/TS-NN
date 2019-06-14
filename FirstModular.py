# Todoes :
'''



'''





import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import tslearn.datasets

from time import time
from torchvision import datasets, transforms
from torch import nn, optim



class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, series, labels):
        'Initialization'
        self.labels = labels
        self.series = series

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.series)

  def __getitem__(self, index):
        'Generates one sample of data'
        return self.series[index],self.labels[index]





class convTwoLayer(torch.nn.Module):
    def __init__(self, input_size,hidden_size,output_size, kernel_size):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(convTwoLayer, self).__init__()
        self.input_size = input_size
        self.output_cha = 20


        self.relu = torch.nn.ReLU()

        self.conv1d  = torch.nn.Conv1d(1,self.output_cha,kernel_size)
        self.linear1 = torch.nn.Linear(139*self.output_cha,hidden_size[0])
        self.linear2 = torch.nn.Linear(hidden_size[0],hidden_size[1])
        self.linear3 = torch.nn.Linear(hidden_size[1],output_size)
        self.maxi = torch.nn.LogSoftmax()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        res1 = self.conv1d(x)
        res2 = self.relu(self.linear1(res1.view(res1.shape[0],-1)))
        res3 = self.relu(self.linear2(res2))
        res4 = (self.linear3(res3))
        res  = self.maxi(res4)

        return res


    def fit(self,data,epochs,optimizer,criterion):
        for e in range(epochs):
            running_loss = 0
            for entries, labels in data:
                entries = entries.reshape(entries.shape[0],1,-1).float()
                # Training pass

                optimizer.zero_grad()
                output = model(entries)
                output = output.view(output.shape[0],-1)
                loss = criterion(output, labels.long())

                # This is where the model learns by backpropagating
                loss.backward()

                # And optimizes its weights here
                optimizer.step()

                running_loss += loss.item()
            if e %30 == 0 :print("Epoch {} - Training loss: {}".format(e, running_loss / len(data)))

    def test_acc(self,data):

        loss, all_count = 0, 0
        for entries, labels in data:
            for i in range(len(labels)):
                entry = entries[i].reshape(1,1,self.input_size).float()
                with torch.no_grad():
                    aux = model(entry)

                res = list(torch.exp(aux).numpy()[0])
                pred_label = res.index(max(res))
                true_label = labels.numpy()[i]
                #if true_label == pred_label : print(true_label)
                loss += true_label == pred_label

                all_count += 1
        print("Number Of Images Tested =", all_count)
        print("\nModel Accuracy =", (loss / all_count).item())

input_size = 140
hidden_sizes = [128, 64]
output_size = 5
kernel_size = 2

model = convTwoLayer(input_size,hidden_sizes,output_size,kernel_size)
X_train, y_train, X_test, y_test = tslearn.datasets.UCR_UEA_datasets().load_dataset("ECG5000")

#Training Data
data_tr = Dataset(X_train,y_train-1)
data_r  = torch.utils.data.DataLoader(data_tr, batch_size=24, shuffle=True)

#Test Data
data_te = Dataset(X_test,y_test-1)
data_e  = torch.utils.data.DataLoader(data_te, batch_size=24, shuffle=True)

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.03)

#On regarde le résultat du modèle avant de l'entrainer

model.test_acc(data_e)

epochs = 500
model.fit(data_r,epochs,optimizer,criterion)

print('On passe au test')

model.test_acc(data_e)


'''
model accuracy 128 128 : 0.5542082786560059
'''