from RNNPred import *
import scipy.spatial as sp
import tslearn.metrics as tsmetrics
import seaborn as sns

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

#@jit
def separate(x,y,n):
    '''

    n: upper bound on number of classes, the number of returned valued aswell
    '''
    vals = dict()
    i = 0
    l = [[] for ind in range(n)]
    for e,elm in enumerate(y):
        if elm in vals:
            l[vals[elm]].append(x[e])
        else:
            vals[elm] = i
            i += 1
            l[vals[elm]].append(x[e])
    return(np.array(l))



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
    return(torch.tensor(x.reshape((1,seq_len,1))).float())


@jit
def FSL(tab,csum,true_lb,lb,shape, prnt = False):
    '''
    FEW SHOT LEARNING -PEW PEW, You're learned-
    '''
    loss = 0
    tot = np.zeros(shape)
    tr  = np.zeros(shape)

    for e,ligne in enumerate(tab):
        a = np.zeros(shape)
        for i,x in enumerate(ligne):
            a[lb[i]-1] += x
        a /= csum[e]
        res = np.argmax(a)
        tot[res] += 1
        tr[true_lb[e]-1] +=1
        #print(res, true_lb[e])
        loss += (res+1 != true_lb[e])
    if prnt : print("FSL :\n",tot,tr)
    return(loss/1000)



def testknn(tab,lab,i,k,true_lab,shape,prnt = False):
    loss = 0
    tot = np.zeros(shape)
    tr = np.zeros(shape)
    for e,ligne in enumerate(tab):
        a = np.zeros(shape)
        idx = np.argpartition(ligne,min(k,i))
        labels = np.take(lab,idx)
        for x in labels:
            a[x-1] +=1
        res = np.argmax(a)
        tot[res] += 1
        tr[true_lab[e] - 1] += 1
        #print(res, true_lab[e])
        loss += (res +1 != true_lab[e])
    if prnt : print("DTW-KNN: \n",tot,tr)
    return(loss/1000)

@jit
def hidden(ax,model):
    return [(model.forward(transform(x), True)).numpy() for x in ax]


def main(n,pas,xtr,xte,ytr,yte,rep = 1, show=True):
    '''
        Moyen plus random pour choisir data/lb
        sauter un peu pour le nbr d'elements de x_train
    '''
    # Initialisation du RNN + entrainement :
    model = code()
    st = time()
    shape = 4
    l = np.zeros(n-1)
    ldtw = np.zeros(n-1)
    lt = np.zeros(n-1)
    ldtwt = np.zeros(n-1)
    print(xte.shape,xtr.shape,yte.shape,ytr.shape)
    with torch.no_grad():
        hiddens_train = hidden(xtr[:, start:end, :],model)
        hiddens_test  = hidden(xte[:, start:end, :],model)


        #On reprend la partie Normale:
        for i in range(rep):
            np.random.shuffle(hiddens_train)
            tabc = np.exp(1 - tab_dist(hiddens_test,hiddens_train,sp.distance.cosine))
            csum = np.cumsum(tabc,axis=1)
            tab = tab_dist(xte, xtr, sp.distance.euclidean)

            for j in range(1,n):
                i = pas*j
                #print(np.bincount(ytr[:i]))
                l[j-1]    = FSL(tabc[:,:i],csum[:,i], yte, ytr, shape)
                ldtw[j-1] = testknn(tab[:,:i],ytr,i-1,5,y_test,shape)
                # Partie ou on regarde les résultats a la fin
            plt.figure()
            plt.plot(range(n - 1), l, label = "Attention kernel method")
            plt.plot(range(n - 1), ldtw - 0.05, label = "KNN with TW")
            plt.legend(loc = 'best')
            plt.xlabel("Number of data")
            plt.ylabel("Percentage of wrong classifications")
            plt.title("0-1 Loss in percentage for Classification")
            plt.show()
            lt += l
            ldtwt += ldtw

        lt /= (rep)
        ldtwt /= (rep)
        if show:
            plt.figure()
            plt.plot(range(n - 1), lt, label = "Attention kernel method")
            plt.plot(range(n - 1), ldtwt, label = "KNN with TW")
            plt.legend(loc = 'best')
            plt.xlabel("Number of data")
            plt.ylabel("Percentage of wrong classifications")
            plt.title("0-1 Loss in percentage for Classification")
            plt.show()

    ft = time() - st
    print("Fin en {} min et {} sec".format(ft // 60, ft % 60))

    if rep == 1:
        # Partie ou on dessine en 2d les Hidden
        elems = separate(TSNE(n_components = 2).fit_transform(hiddens_train), ytr, 3)
        plt.figure()
        for i, coordi in enumerate(elems):
            plt.scatter(coordi[:, 0], coordi[:, 1], label = "{}-th Label".format(i))
        plt.legend(loc = 'best')
        plt.show()
        #Partie ou on regarde les résultats a la fin
        plt.figure()
        plt.plot(range(n-1),l,label = "Attention kernel method")
        plt.plot(range(n-1),ldtw-0.05, label = "KNN with TW")
        plt.legend(loc = 'best')
        plt.xlabel("Number of data")
        plt.ylabel("Percentage of wrong classifications")
        plt.title("0-1 Loss in percentage for Classification")
        plt.show()

    return(lt[-1],model)


xsp,ysp = shuffle_in_unison(xsp,ysp)
ret = main(60,5,xsp,X_test,ysp,y_test,20)


def swigitty(lng):

    classacc = np.zeros(lng-1)
    errtab = np.zeros((lng-1,4500))
    pas_main = 5
    pas = 5
    moyen = 20
    xtr, xte,ytr,yte = xsp,X_test,ysp,y_test

    for j in range(1,lng):
        i = lng*pas
        hidden_sizes[0] = i
        res,model = main(seql,pas_main,xtr,xte,ytr,yte,rep = moyen, show=False)
        lt[j] = res
        l[i] += model.test_tab()

    sns.set(style = "darkgrid")
    "FAUT QUE JE TROUVE COMMENT plot CECI"




'''
Done all    

'''
