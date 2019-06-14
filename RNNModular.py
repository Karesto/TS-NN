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





class RNNMod(torch.nn.Module):
    def __init__(self, ins,hs,os,ocha,length = 140,lstms = 1):
        '''
        length = longueur des time series


        '''
        super(RNNMod, self).__init__()
        self.ins = ins
        self.length = length

        self.relu = torch.nn.ReLU()

        self.rnn = nn.LSTM(1, hs[0], lstms, batch_first = True)
        self.linear1 = torch.nn.Linear(hs[0],hs[1])
        self.linear2 = torch.nn.Linear(hs[1],os)
        self.maxi = torch.nn.LogSoftmax()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        a = x.shape
        hidden = None
        for i in range(self.length):
            out, hidden = self.rnn(x[:,:,i].view(a[0],a[1],1),hidden)
        res1 = self.relu(self.linear1(out.view(out.shape[0],-1)))
        res2 = (self.linear2(res1))
        res  = self.maxi(res2)

        return res


    def fit(self,data,epochs,optimizer,criterion):
        time0 =  time()
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
            if e %3 == 0 :print("Epoch {} - Training loss: {}".format(e, running_loss / len(data)))
        timet = time() - time0
        print("Fin de l'étape d'apprentissage en {} min et {} sec".format(timet//60, timet%60))


    def test_acc(self,data):
        guesses = np.zeros(5)
        total_labels = np.zeros(5)
        loss, all_count = 0, 0
        for entries, labels in data:
            for i in range(len(labels)):
                entry = entries[i].reshape(1,1,self.ins).float()
                with torch.no_grad():
                    aux = model(entry)

                res = list(torch.exp(aux).numpy()[0])
                pred_label = res.index(max(res))
                true_label = labels.numpy()[i]
                guesses[pred_label] += 1
                total_labels[true_label] += 1

                #if true_label == pred_label : print(true_label)
                loss += true_label == pred_label

                all_count += 1
        print("Number Of Images Tested =", all_count)
        print("Model Accuracy =", (loss / all_count).item())
        print("Guesses overall : ", guesses)
        print("Actual numbers :", total_labels)

input_size = 140
hidden_sizes = [128, 128]
output_size = 5
output_channels = 20
hdim = 1

model = RNNMod(input_size,hidden_sizes,output_size,output_channels)
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

#model.test_acc(data_e)

epochs = 10
model.fit(data_r,epochs,optimizer,criterion)

print('On passe au test')

model.test_acc(data_e)


'''
Apparment -> The more LSTM the less it recognizes the less represented elements


model accuracy lstm = 2, epochs = 30 : 0.5837 --> 0.8977
    Avec plus d'epochs on devrait avoir de meilleurs résultats car le gradient n'est pas encore fixe (mais proche)
    ne prend générallement que 1 ou 2, 3 4 et 5 ont des valeurs extrêmement petites (à voir)
    stabilise a epoch = 50 approx, 
    si on attend jusqu'a epoch = 100 : on a 0.8993 très peu d'amélioration pour pas mal d'attente.
    


model accuracy lstm = 4, epochs = 30 : 
    0.5837 --> 0.5837 (ne reconnait que les 2) (unchanged, might be because issues)
    Avec 30 epochs, la loss descend toujours sur le train set, environs 0.9 a la fin, mais le temps d'execution est déja si lent
    La loss change bcp a 70 epochs, a 100 epochs on a finallement 0.8986 de loss qui reconnait que 1 et 2


lstm = 3 : risque se stabilise a epoch 70 vers 0.35


LSTM = 1 : se stabilise assez tot , vers epoch = 40, a 0.902 de réussite après 100 epochs, toujours 0 pour les 2 a 4
un test avec epoch = 10 donne 0.845 et que pour les 2 premiers



'''