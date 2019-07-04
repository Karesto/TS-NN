from RNNPred import *

import scipy.spatial as sp


def distance(a,b):
    return np.linalg.norm((a-b))**2


def shuffle_in_unison(a, b):
    n_elem = a.shape[0]
    indeces = np.random.choice(n_elem, size = n_elem, replace = False)
    return a[indeces], b[indeces]


def knn(k,data,lb,x):
    res = np.array([distance(x,y) for y in data])
    x1, y1 = zip(*sorted(zip(res, lb)))
    x1, y1 = np.array(x1),np.array(y1)
    return(np.floor(np.mean(y1[0:k]) + 0.5))


def cosdis(x,y):
    '''
    Cosine Distance. To do : it all
    '''
    return sp.distance.cosine(x,y)

def transform(x):
    '''
    Transforme l'entrée x en une entrée correcte pour le modèle RNN
    forme à la sortie de transform :
    batch size,length,1
    '''
    return(torch.tensor(x.reshape((1,80,1))).float())

def FSL(data,lb,x,model,shape):
    '''
    FEW SHOT LEARNING -PEW PEW, You're learned-

    We will consider data is transformed (aka we can make it go through our model directly)
    '''
    n = len(data)
    s,tot = np.zeros(shape),0
    t = (model.forward(transform(x), True)[1]).numpy()
    for i in range(n):
        calc = cosdis((model.forward(transform(data[i]), True)[1]).numpy(), t)
        s[lb[i]] += calc
        tot += calc

    return(np.argmax(s/tot))


def test(data_test,true_lb,data,lb,model,shape):
    loss = 0
    for e,x in enumerate(data_test):
        loss += FSL(data,lb,x,model,shape) != true_lb[e]
    return(loss)





def main(maxl):
    '''
        Moyen plus random pour choisir data/lb
        sauter un peu pour le nbr d'elements de x_train
        shuffle X_train, Y_train
        '''
    # Initialisation du RNN + entrainement :
    code()

    shape = 5
    n = maxl
    l=[0]*(n-1)
    print(X_test.shape,y_test.shape)
    with torch.no_grad():
        for j in range(1,n):
            i = 5*j
            st = time()
            print(i)
            data = X_train[:i,start:end,:]
            lb = y_train[:i]
            l[j-1] = test(X_test[:700,start:end,:], y_test[:700], data, lb, model, shape)
            ft = time() - st
            print("Fin en {} min et {} sec".format(ft // 60, ft % 60))

    plt.plot(range(1,n),l)


X_train, y_train = shuffle_in_unison(X_train,y_train)
X_test, y_test = shuffle_in_unison(X_test,y_test)

main(10)