# Todoes :
'''
Voire des modèles simples (Time warmping + KNN)
donner la classe en meme temps
apprendre a plusieurs endroits de la série

----------------------
Faire une moyenne pour le KNN
Correlation avec la prédiction (en fonction de h)
Tester plusieurs h



'''

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import tslearn.datasets
import seaborn as sns

from sklearn.manifold import TSNE

from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from numba import jit


#################### Fonctions Diverses ###############################

@jit
def shuffle_in_unison(a, b, shuffle_arg = False):
    n_elem = len(a)
    indeces = np.random.choice(n_elem, size = n_elem, replace = False)
    if shuffle_arg:
        return a[indeces], b[indeces], indeces
    else :
        return a[indeces], b[indeces]


@jit
def cleanse(x,y):
    l = []
    for e,elm in enumerate(y):
        if elm == 3 or elm == 5: l.append(e)
    return np.delete(x,l,axis =0), np.delete(y,l)


@jit
def repartition(x,y):
    l = [ [], [], [] ]
    for e,elm in enumerate(y):
        if elm == 4 : l[elm - 2].append(e)
        else : l[elm - 1].append(e)
    n = min(len(l[2]),100)
    l=np.array(l)
    print(n)
    a = np.concatenate((x[l[0][:n]],x[l[1][:n]],x[l[2][:n]]),axis =0)
    b = np.concatenate((y[l[0][:n]],y[l[1][:n]],y[l[2][:n]]))
    return(a,b)



################################# Dataset #################################
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

    def get_random(self):
        'Generates a random element of the dataset'
        n = self.__len__()
        i = np.random.randint(n)
        return self.series[i], self.labels[i], i

################################# RNNPred #################################

class RNNPred(torch.nn.Module):
    def __init__(self, ins, hs, os,length = 140, lstms = 1):
        '''
        length = longueur des time series


        '''
        super(RNNPred, self).__init__()
        self.ins = ins
        self.length = length

        self.relu = torch.nn.ReLU()

        self.rnn = nn.LSTM(1, hs[0], lstms, batch_first = True)
        self.linear1 = torch.nn.Linear(ins*hs[0], hs[0])
        self.linear2 = torch.nn.Linear(hs[0], hs[2])
        self.linear3 = torch.nn.Linear(hs[2], os)

    def forward(self, x, hidd = False):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        a = x.shape
        x = x.view(a[0],a[2],a[1])

        hidden = None
        for i in range(a[1]):
            out, hidden = self.rnn(x[:, :, i:i+1], hidden)
        out = out.contiguous()
        res1 = self.relu(self.linear1(out.view(out.shape[0], -1)))
        res2 = self.relu(self.linear2(res1))
        res  = self.linear3(res2)
        if hidd:
            return hidden[0].view(-1)
        return res

    def fit(self, data, epochs, optimizer, criterion):
        time0 = time()
        for e in range(epochs):
            running_loss = 0
            for entries, labels in data:
                entries = entries.reshape(entries.shape[0], 1, -1).float()
                # Training pass

                optimizer.zero_grad()
                output = self(entries)
                output = output.view(output.shape[0], -1, 1)
                loss = criterion(output, labels.float())

                # This is where the model learns by backpropagating
                loss.backward()

                # And optimizes its weights here
                optimizer.step()

                running_loss += loss.item()
            if e % 5 == 0: print("Epoch {} - Training loss: {}".format(e, running_loss / len(data)))
        timet = time() - time0
        print("Fin de l'étape d'apprentissage en {} min et {} sec".format(timet // 60, timet % 60))

    def test_acc(self, data, perc):
        loss, zoloss, all_count = 0, 0, 0
        for entries, labels in data:
            for i in range(len(labels)):
                entry = entries[i].reshape(1, 1, self.ins).float()
                with torch.no_grad():
                    pred = self(entry)

                true = labels.numpy()[i]
                # if true_label == pred_label : print(true_label)
                loss += np.abs(true.reshape(5) - pred.view(5).numpy())
                #zoloss += np.abs(true-pred.item()) > epsi.item()

                all_count += 1
        print("Number Of Images Tested =", all_count)
        print("Model Accuracy (L1-loss) =", (loss / all_count))


    def test_tab(self,data):
        l = [0]*4500
        for entries, labels in data:
            j = 0
            for i in range(len(labels)):
                entry = entries[i].reshape(1, 1, self.ins).float()
                with torch.no_grad():
                    pred = self(entry)

                true = labels.numpy()[i]
                l[j] = np.sum(np.abs(true.reshape(5) - pred.view(5).numpy()))

                j += 1
        return np.array(l)/4500

    def test_img(self,x, x_true,i):

        n = len(x) + len(x_true)
        y1   = np.append(x,x_true)
        res  = (self.forward(torch.tensor(x).view(1,self.ins,-1).float())).detach().numpy()
        y2   = np.append(x,res)
        n    = len(y1)
        loss = (np.abs(res.reshape(5)-x_true.reshape(5)))
        abc  = range(n)
        plt.plot(abc, y1,)
        plt.plot(abc, y2,)
        plt.title("The {}_th element ins the shuffled array. \n Loss is : {}".format(i,np.sum(loss)))
        plt.show()


    def sing_loss(self,entry,true):
        with torch.no_grad():
            entry = entry.reshape(1, 1, self.ins).float()
            pred = model(entry)
            loss = np.abs(true.numpy().reshape(5) - pred.numpy().reshape(5))
        return loss

################################# RNNClass ################################

class RNNClass(torch.nn.Module):
    def __init__(self, ins, hs, os, length = 140, lstms = 1):
        '''
        length = longueur des time series
        '''
        super(RNNClass, self).__init__()
        self.ins = ins
        self.length = length
        self.crit = nn.NLLLoss()
        self.relu = torch.nn.ReLU()

        self.rnn = nn.LSTM(1, hs[0], lstms, batch_first = True)
        self.linear1 = torch.nn.Linear(ins*hs[0], hs[0])
        self.linear2 = torch.nn.Linear(hs[0], 5)
        self.maxi = torch.nn.LogSoftmax(dim=1)

    def forward(self, x, hidd = False):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        a = x.shape
        x = x.view(a[0],a[2],a[1])


        hidden = None
        for i in range(a[1]):
            out, hidden = self.rnn(x[:, :, i:i+1], hidden)
        out = out.contiguous()
        res1 = self.relu(self.linear1(out.view(out.shape[0], -1)))
        res2 = (self.linear2(res1))
        res = self.maxi(res2)
        return res

    def fit(self, data, epochs, optimizer, prnt = False ):
        time0 = time()
        for e in range(epochs):
            running_loss = 0
            for entries, labels in data:
                entries = entries.reshape(entries.shape[0], 1, -1).float()
                # Training pass

                optimizer.zero_grad()
                output = self.forward(entries)
                output = output.view(output.shape[0], -1)
                loss = self.crit(output, labels.long())

                # This is where the model learns by backpropagating
                loss.backward()

                # And optimizes its weights here
                optimizer.step()

                running_loss += loss.item()
            if e % 5 == 0 and prnt: print("Epoch {} - Training loss: {}".format(e, running_loss / len(data)))
        timet = time() - time0
        print("Fin de l'étape d'apprentissage en {} min et {} sec".format(timet // 60, timet % 60))

    def test_acc(self, data, perc, prnt = False):
        guesses = np.zeros(5)
        total_labels = np.zeros(5)
        loss, all_count = 0, 0
        for entries, labels in data:
            for i in range(len(labels)):
                entry = entries[i].reshape(1,1,self.ins).float()
                with torch.no_grad():
                    aux = self.forward(entry)

                res = list(torch.exp(aux).numpy()[0])
                pred_label = res.index(max(res))
                true_label = labels.numpy()[i]
                guesses[pred_label] += 1
                total_labels[true_label] += 1

                #if true_label == pred_label : print(true_label)
                loss += true_label != pred_label

                all_count += 1
        if prnt :
            print("Number Of Images Tested =", all_count)
            print("Model Accuracy =", 1 - (loss / all_count).item())
            print("Guesses overall : ", guesses)
            print("Actual numbers :", total_labels)

        return (loss / all_count).item()


################################# RNNDBPred #################################

class RNNdoubleheadPred(torch.nn.Module):
    def __init__(self, ins, hs, os,modul, length = 140, lstms = 1):
        '''
        length = longueur des time series
        '''
        super(RNNdoubleheadPred, self).__init__()
        self.ins = ins
        self.os  = os
        self.length = length
        self.relu = torch.nn.ReLU()
        self.crit1 = nn.MSELoss()
        self.crit2 = nn.NLLLoss()

        self.rnn = nn.LSTM(1, hs[0], lstms, batch_first = True)
        self.linear1 = torch.nn.Linear(ins*hs[0], hs[0])
        self.linear2 = torch.nn.Linear(hs[0], hs[2])
        self.linear3 = torch.nn.Linear(hs[2], os)
        self.linear4 = torch.nn.Linear(hs[2],5)
        self.modul = modul
        self.maxi = torch.nn.LogSoftmax(dim = 1)

    def forward(self, x, sel = True ,hidd = False):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        a = x.shape
        x = x.view(a[0],a[2],a[1])


        hidden = None
        for i in range(a[1]):
            out, hidden = self.rnn(x[:, :, i:i+1], hidden)
        if hidd:
            return hidden[0].view(-1)
        out = out.contiguous()
        res1 = self.relu(self.linear1(out.view(out.shape[0], -1)))
        res2 = (self.linear2(res1))


        if sel :
            res3 = self.linear3(res2)
            return res3
        else :
            res = self.maxi(self.linear4(res2))
            return res


    def fit(self, data, epochs, opti, prnt = False):
        time0 = time()
        i = 0
        for e in range(epochs):

            running_loss = 0
            for entries, labels in data:
                if i % self.modul != 0 :
                    labl = labels[:,:self.os,0]

                    entries = entries.reshape(entries.shape[0], 1, -1).float()
                    # Training pass

                    opti.zero_grad()
                    output = self(entries)
                    output = output.view(output.shape[0], -1)
                    loss = self.crit1(output, labl.float())

                    # This is where the model learns by backpropagating
                    loss.backward()

                    # And optimizes its weights here
                    opti.step()

                    running_loss += loss.item()
                else:
                    labl = labels[:,self.os,0]

                    entries = entries.reshape(entries.shape[0], 1, -1).float()
                    # Training pass

                    opti.zero_grad()
                    output = self(entries, sel = False)
                    output = output.view(output.shape[0], -1)
                    loss = self.crit2(output, labl.long())

                    # This is where the model learns by backpropagating
                    loss.backward()

                    # And optimizes its weights here
                    opti.step()

                    running_loss += loss.item()


            if e % 5 == 0 and prnt: print("Epoch {} - Training loss: {}".format(e, running_loss / len(data)))
        timet = time() - time0
        print("Fin de l'étape d'apprentissage en {} min et {} sec".format(timet // 60, timet % 60))

    def test_acc(self, data, perc ,prnt = False):
        guesses = np.zeros(5)
        total_labels = np.zeros(5)
        loss, all_count = 0, 0
        for entries, labels in data:
            for i in range(len(labels)):
                entry = entries[i].reshape(1,1,self.ins).float()
                with torch.no_grad():
                    aux = self.forward(entry)

                res = list(torch.exp(aux).numpy()[0])
                pred_label = res.index(max(res))
                true_label = labels.numpy()[i]
                guesses[pred_label] += 1
                total_labels[true_label] += 1

                #if true_label == pred_label : print(true_label)
                loss += true_label != pred_label

                all_count += 1
        if prnt:
            print("Number Of Images Tested =", all_count)
            print("Model Accuracy =", 1 - (loss / all_count).item())
            print("Guesses overall : ", guesses)
            print("Actual numbers :", total_labels)



################################# RNNDBClass #################################

class RNNdoubleheadClass(torch.nn.Module):
    def __init__(self, ins, hs, os,modul, length = 140, lstms = 1):
        '''
        length = longueur des time series
        '''
        super(RNNdoubleheadClass, self).__init__()
        self.ins = ins
        self.os = os
        self.length = length
        self.relu  = torch.nn.ReLU()
        self.crit1 = nn.NLLLoss()
        self.crit2 = nn.MSELoss()
        self.modul   = modul


        self.rnn = nn.LSTM(1, hs[0], lstms, batch_first = True)
        self.linear1 = torch.nn.Linear(ins*hs[0], hs[0])
        self.linear2 = torch.nn.Linear(hs[0], hs[2])
        self.linear3 = torch.nn.Linear(hs[2],os)
        self.linear4 = torch.nn.Linear(hs[2],5)
        self.maxi = torch.nn.LogSoftmax(dim =1)

    def forward(self, x, sel = True ,hidd = False):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        a = x.shape
        x = x.view(a[0],a[2],a[1])


        hidden = None
        for i in range(a[1]):
            out, hidden = self.rnn(x[:, :, i:i+1], hidden)
        out = out.contiguous()
        res1 = self.relu(self.linear1(out.view(out.shape[0], -1)))
        res2 = (self.linear2(res1))

        if sel :
            res = self.maxi(self.linear4(res2))
            return res
        else :
            res3 = self.linear3(res2)
            return res3

    def fit(self, data, epochs, opti, prnt = False):
        time0 = time()
        i = 0
        for e in range(epochs):

            running_loss = 0
            for entries, labels in data:
                if i % self.modul != 0 :
                    entries = entries.reshape(entries.shape[0], 1, -1).float()
                    # Training pass
                    labl = labels[:,self.os,0]
                    opti.zero_grad()
                    output = self.forward(entries)
                    output = output.view(output.shape[0], -1)
                    loss = self.crit1(output, labl.long())

                    # This is where the model learns by backpropagating
                    loss.backward()

                    # And optimizes its weights here
                    opti.step()

                    running_loss += loss.item()
                else:
                    labl = labels[:,:self.os,0]
                    entries = entries.reshape(entries.shape[0], 1, -1).float()
                    # Training pass

                    opti.zero_grad()
                    output = self(entries, sel = False)
                    output = output.view(output.shape[0], -1)
                    loss = self.crit2(output, labl.float())

                    # This is where the model learns by backpropagating
                    loss.backward()

                    # And optimizes its weights here
                    opti.step()

                    running_loss += loss.item()
                i+= 1

            if e % 5 == 0 and prnt: print("Epoch {} - Training loss: {}".format(e, running_loss / len(data)))
        timet = time() - time0
        print("Fin de l'étape d'apprentissage en {} min et {} sec".format(timet // 60, timet % 60))

    def test_acc(self, data, perc, prnt = False):
        guesses = np.zeros(5)
        total_labels = np.zeros(5)
        loss, all_count = 0, 0
        for entries, labels in data:
            for i in range(len(labels)):
                entry = entries[i].reshape(1, 1, self.ins).float()
                with torch.no_grad():
                    aux = self.forward(entry)

                res = list(torch.exp(aux).numpy()[0])
                pred_label = res.index(max(res))
                true_label = labels.numpy()[i]
                guesses[pred_label] += 1
                total_labels[true_label] += 1

                # if true_label == pred_label : print(true_label)
                loss += true_label == pred_label

                all_count += 1
        if prnt:
            print("Number Of Images Tested =", all_count)
            print("Model Accuracy =", (loss / all_count).item())
            print("Guesses overall : ", guesses)
            print("Actual numbers :", total_labels)

        return (loss / all_count).item()



#Data Coding

X_train, y_train, X_test, y_test = tslearn.datasets.UCR_UEA_datasets().load_dataset("ECG5000")

Cleanup = False
plotter = False
if Cleanup:

    a,b = cleanse(np.append(X_train,X_test,axis = 0),np.append(y_train,y_test))
    print(a.shape,b.shape)
    a,b = shuffle_in_unison(a,b)
    xe,xr  = a[:1000], a[1000:]
    ye,yr  = b[:1000], b[1000:]


    xsp,ysp = repartition(xr,yr)

    X_train, X_test, y_train, y_test = xr, xe, yr,ye
else:
    xsp = X_train
    ysp = y_train

if plotter:
    res1 = np.bincount(y_train)[1:]
    res2 = np.bincount(y_test)[1:]

    plt.figure()
    ntr = len(y_train)
    pertr = res1*100/ntr
    labtr = ["{}%".format(i) for i in pertr]
    sns.set(style = "darkgrid")
    plt.bar(range(1,len(res1)+1),res1)
    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(res1):
        plt.text(xlocs[i+1]-0.2, v + 0.01, labtr[i])
    plt.title("Repartition of Train Data")
    plt.xlabel("Labels")
    plt.ylabel("Number of elements")
    #plt.legend(loc = 'best')

    nte = len(y_test)
    pertr = np.floor(res2 * 10000 / nte)/100
    labtr = ["{}%".format(i) for i in pertr]
    plt.figure()
    sns.set(style = "darkgrid")
    plt.bar(range(1, len(res2)+1), res2)
    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(res2):
        plt.text(xlocs[i + 1] - 0.24, v + 0.01, labtr[i])
    plt.title("Repartition of Test Data")
    plt.xlabel("Labels")
    plt.ylabel("Number of elements")
    #plt.legend(loc = 'best')

n = X_train.shape[1]
testsize = X_test.shape[0]


length  = 5
start   = 0
seq_len = 135
end     = start + seq_len

############################## Data for RNNPred #########################

# Training Data
data_tr = Dataset(X_train[:,start:end,:], X_train[:,end:end + length,:])
data_r = torch.utils.data.DataLoader(data_tr, batch_size = 24, shuffle = True)

# Test Data
data_te = Dataset(X_test[:,start:end,:], X_test[:,end:end + length,:])
data_e = torch.utils.data.DataLoader(data_te, batch_size = 24, shuffle = False)

############################## Data for RNNClass #########################

# Training Data
data_tr = Dataset(X_train,y_train-1)
data_r_class = torch.utils.data.DataLoader(data_tr, batch_size = 24, shuffle = True)

# Test Data
data_te = Dataset(X_test,y_test-1)
data_e_class = torch.utils.data.DataLoader(data_te, batch_size = 24, shuffle = False)

############################## Data for DBhead #########################

# Training Data
au1 = X_train[:,end:end + length,:]
au2 = (y_train-1).reshape(y_train.shape[0],1,1)
au  = np.append(au1,au2,axis=1)

data_tr = Dataset(X_train[:,start:end,:],au)
data_r_db = torch.utils.data.DataLoader(data_tr, batch_size = 24, shuffle = True)

# Test Data

au1 = X_test[:,end:end + length,:]
au2 = (y_test-1).reshape(y_test.shape[0],1,1)
au = np.append(au1,au2,axis=1)

data_te = Dataset(X_test[:,start:end,:],y_test-1)
data_e_db = torch.utils.data.DataLoader(data_te, batch_size = 24, shuffle = False)


'''
utiliser les implems


'''




#Parameters

input_size = seq_len
hidden_sizes = [32, 128, 128]
output_size = length
output_channels = 20
hdim = 1



# On regarde le résultat du modèle avant de l'entrainer
def test(model,data):
    return model.test_acc(data,0.1)

def train(epochs,model,optimizer,criterion):
    model.fit(data_r, epochs, optimizer, criterion)


def ts_test(model):
    x, xt, i = data_te.get_random()
    print(y_test[i])
    model.test_img(x, xt, i)

def histo(num_class, longe, model):
    '''
    à modifier pour une version plus générale pour une sortie différente de 5/ plus que 5 types
    Memes axes , ajouter des titres et descriptions pour que ce soit plus compréhensible
    '''
    indx = 0
    tots = np.zeros(num_class)
    totloss = np.zeros((num_class,longe))
    for entries, labels in data_e:
        for i in range(len(labels)):
            totloss[y_test[indx]-1] += model.sing_loss(entries[i],labels[i])
            tots[y_test[indx]-1] += 1
            indx +=1
    fig,ax = plt.subplots(1,num_class,sharey =  True)

    fig.add_subplot(111, frameon = False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor = 'none', top = False, bottom = False, left = False, right = False)
    plt.xlabel("common X")
    plt.ylabel("common Y")


    for i in range(longe):
        totloss[:,i] /= tots
    for i in range(num_class):
        ax[i].bar(np.arange(longe),totloss[i])
        ax[i].set_title("Classe {}".format(i))

    plt.show()

    return totloss,tots

def code():

    model = RNNPred(input_size, hidden_sizes, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.03)

    #test(model)
    model.fit(data_r, 50, optimizer, criterion)
    test(model,data_e)
    #totloss,tots = histo(5,5,model)
    #ts_test(model)
    return model

def codeclass(ft = False):
    model = RNNClass(140, hidden_sizes, 5)
    optimizer = optim.SGD(model.parameters(), lr = 0.03)
    if ft : model.fit(data_r_class, 50, optimizer)
    #test(model,data_e_class)
    return model

def codedbheadclass(ft= False):
    model = RNNdoubleheadClass(input_size, hidden_sizes, output_size, 10)
    optimizer = optim.SGD(model.parameters(), lr = 0.03)
    if ft : model.fit(data_r_db, 50, optimizer)
    return model

def codedbheadpred():
    model = RNNdoubleheadPred(input_size, hidden_sizes, output_size, 10)
    optimizer = optim.SGD(model.parameters(), lr = 0.03)
    model.fit(data_r_db, 50, optimizer)
    return model




'''
--> Réseau double sortie

-----------------------------------------------------

OLD COMMENTS :


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

--------------------------------------------------------------------------------------

On peut maintenant tester pour voir ce que cela donne, et les résultats sont plutôt convaincants.
Pour l'instant on est sur de l'apprentissage sur toute la série temporelle. On va essayer de prendre des petits bouts maintenant.


---------------------------------------------------------------------------------------








'''
