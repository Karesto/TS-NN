from RNNPred import *
import scipy.spatial as sp
import tslearn.metrics as tsmetrics


def distance(a,b):
    return tsmetrics.dtw(a,b)

@jit
def tab_dist(a,b,dist):
    n1,n2 = len(a), len(b)
    res = np.zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            res[i][j] = dist(a[i],b[j])
    return res

def knn(k,data,lb,x,tab):
    '''
    Supposed to do KNN but is slow (recomputing distances) AND not correct (as it uses mean and not buckets)
    '''
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


@jit
def FSL(tab,csum,true_lb,lb,model,shape):
    '''
    FEW SHOT LEARNING -PEW PEW, You're learned-
    '''
    loss = 0
    for e,ligne in enumerate(tab):
        a = np.zeros(5)
        for i,x in enumerate(ligne):
            a[lb[i]-1] += x
        a /= csum[e]
        res = np.argmax(a)
        loss += (res+1 != true_lb[e])
    return(loss/1000)



def testknn(tab,lab,i,k,true_lab):
    loss = 0
    for e,ligne in enumerate(tab):
        a = np.zeros(5)
        idx = np.argpartition(ligne,min(k,i))
        labels = np.take(lab,idx)
        for x in labels:
            a[x-1] +=1
        res = np.argmax(a)
        loss += (res +1 != true_lab[e])
    return(loss/1000)

@jit
def hidden(ax):
    return [(model.forward(transform(x), True)).numpy() for x in ax]

def main(maxl,xtr,xte,ytr,yte):
    '''
        Moyen plus random pour choisir data/lb
        sauter un peu pour le nbr d'elements de x_train
    '''
    # Initialisation du RNN + entrainement :
    code()
    st = time()
    shape = 5
    n = maxl
    l = np.zeros(n-1)
    ldtw = np.zeros(n-1)
    print(xte.shape,xtr.shape,yte.shape,ytr.shape)
    with torch.no_grad():
        hiddens_train = hidden(xtr[:, start:end, :])
        hiddens_test  = hidden(xte[:, start:end, :])
        tabc = np.exp(1 - tab_dist(hiddens_test,hiddens_train,sp.distance.cosine))
        csum = np.cumsum(tabc,axis=1)
        tab = tab_dist(xte, xtr, sp.distance.euclidean)

        print("entrée de la boucle")
        for j in range(1,n):
            i = 3*j
            l[j-1]    = FSL(tabc[:,:i],csum[:,i], yte, ytr, model, shape)
            ldtw[j-1] = testknn(tab[:,:i],y_train,i-1,5,y_test)

    ft = time() - st
    print("Fin en {} min et {} sec".format(ft // 60, ft % 60))

    plt.figure()
    plt.plot(range(n-1),l,label = "Attention kernel method")
    plt.plot(range(n-1),ldtw-0.05, label = "KNN with TW")
    plt.legend(loc = 'best')
    plt.xlabel("Number of data")
    plt.ylabel("Percentage of wrong classifications")
    plt.title("0-1 Loss in percentage for Classification")
    plt.show()




xsp,ysp = shuffle_in_unison(xsp,ysp)
main(99,xsp,X_test,ysp,y_test)


'''
Visualisation dasn l'espace caché
Calculer les cosdis avant
mettre les exponentielle et enlever 1-
mettre une moyenne sur différent tirages 


Prendre son temps pour réarranger les fonctions utiles dans des modules


'''
