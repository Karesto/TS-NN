# Todoes :
'''
Voire des modèles simples
donner la classe en meme temps
apprendre a plusieurs endroits de la série

----------------------

ht -> dans un espace et ranger selon la classification




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
        return self.series[index], self.labels[index]


class RNNPred(torch.nn.Module):
    def __init__(self, ins, hs, os, ocha, length = 140, lstms = 1):
        '''
        length = longueur des time series


        '''
        super(RNNPred, self).__init__()
        self.ins = ins
        self.length = length

        self.relu = torch.nn.ReLU()

        #self.rnn = nn.LSTM(1, hs[0], lstms, batch_first = True)
        self.linear1 = torch.nn.Linear(ins, hs[0])
        self.linear2 = torch.nn.Linear(hs[0], hs[2])
        self.linear3 = torch.nn.Linear(hs[2], os)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        a = x.shape
        print(a)
        hidden = None
        #for i in range(a[2]):
            #out, hidden = self.rnn(x[:, :, i].view(a[0], a[1], 1), hidden)
        res1 = self.relu(self.linear1(x.view(x.shape[0], -1)))
        res2 = self.relu(self.linear2(res1))
        res  = self.linear3(res2)

        return res

    def fit(self, data, epochs, optimizer, criterion):
        time0 = time()
        for e in range(epochs):
            running_loss = 0
            for entries, labels in data:
                entries = entries.reshape(entries.shape[0], 1, -1).float()
                # Training pass

                optimizer.zero_grad()
                output = model(entries)
                output = output.view(output.shape[0], -1, 1)
                loss = criterion(output, labels.float())

                # This is where the model learns by backpropagating
                loss.backward()

                # And optimizes its weights here
                optimizer.step()

                running_loss += loss.item()
            if e % 3 == 0: print("Epoch {} - Training loss: {}".format(e, running_loss / len(data)))
        timet = time() - time0
        print("Fin de l'étape d'apprentissage en {} min et {} sec".format(timet // 60, timet % 60))

    def test_acc(self, data, perc):
        loss, zoloss, all_count = 0, 0, 0
        for entries, labels in data:
            for i in range(len(labels)):
                entry = entries[i].reshape(1, 1, self.ins).float()
                with torch.no_grad():
                    pred = model(entry)
                #epsi = torch.max(torch.abs(entry))*perc

                true = labels.numpy()[i]
                #print(true,pred)
                # if true_label == pred_label : print(true_label)
                loss += np.abs(true.reshape(5) - pred.view(5).numpy())
                #zoloss += np.abs(true-pred.item()) > epsi.item()

                all_count += 1
        print("Number Of Images Tested =", all_count)
        print("Model Accuracy (l1-loss) =", (loss / all_count))
        #print("Model Accuracy (0-1 loss with perc = {}) =".format(perc), (zoloss / all_count).item())


input_size = 135
hidden_sizes = [128, 64, 64]
output_size = 5
output_channels = 20
hdim = 1

model = RNNPred(input_size, hidden_sizes, output_size, output_channels)
X_train, y_train, X_test, y_test = tslearn.datasets.UCR_UEA_datasets().load_dataset("ECG5000")

# Training Data
data_tr = Dataset(X_train[:,0:135,:], X_train[:,135:140,:])
data_r = torch.utils.data.DataLoader(data_tr, batch_size = 24, shuffle = True)

# Test Data
data_te = Dataset(X_test[:,0:135,:], X_test[:,135:140,:])
data_e = torch.utils.data.DataLoader(data_te, batch_size = 24, shuffle = True)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.03)

# On regarde le résultat du modèle avant de l'entrainer

model.test_acc(data_e,0.1)

epochs = 75
model.fit(data_r, epochs, optimizer, criterion)

print('On passe au test')

model.test_acc(data_e,0.1)

'''

l1 loss , training pendant 2XX epochs , 5 prédictions, lstm =2, 2 linear (128,128):

[0.20496206 0.31164772 0.4228258  0.54931241 0.67645596]

Plus on séloigne du point le plus proche plus on a du mal a deviner ce qu'il faut faire ce qui est plus ou moins a quoi on s'attends

---------------------
l1 loss , 5 prédictions, lstm =1, 3 linear(128,64,64) :

75 Epochs : Model Accuracy (l1-loss) = [0.32147136 0.48832314 0.65492892 0.78805073 0.89032827]

Stable vers 200 Eporhs : Model Accuracy (l1-loss) = [0.20402207 0.33185193 0.44552198 0.57691824 0.6950413 ]
Pas vraiment mieux

---------------------

epochs = 75, lstm = 0, linear = 3 (128,64,64):

Model Accuracy (l1-loss) = [0.16622388 0.2063745  0.24493237 0.27409063 0.41701113]

On apprend mieux, mais ce n'est toujours pas convaincant.

'''