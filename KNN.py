from RNNPred import *
import scipy.spatial as sp
import tslearn.metrics as tsmetrics
import seaborn as sns
import tslearn.neighbors as tsn


def distance(a,b):
    return tsmetrics.dtw(a,b)

metric = "euclidean"




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
    l = [np.array([]) for ind in range(n)]
    for e,elm in enumerate(y):
        if elm in vals:
            l[vals[elm]] = np.append(l[vals[elm]],[x[e]],axis=0)
        else:
            vals[elm] = i
            i += 1
            l[vals[elm]] = [x[e]]

    return(np.array(l))


@jit
def shuffleaccord(a,p,axis):
    '''
    Shuffles the Array A according to the Permutation p
    '''
    return np.take(a,p,axis)


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
def FSL(tab,csum,lb,true_lb,shape, prnt = False):
    '''
    FEW SHOT LEARNING -PEW PEW, You're learned-
    lb : labels of train
    true_lb : label of test
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
    return(loss/testsize)


def testknn(xtr,ytr,xte,yte,shap,prnt = False):
    print(xtr.shape, ytr.shape, xte.shape , yte.shape)
    #tot = np.zeros(shap)
    #tr = np.zeros(shap)
    clf = tsn.KNeighborsTimeSeriesClassifier(n_neighbors = 1, metric = metric)
    res = clf.fit(xtr,ytr).predict(xte)
    loss = np.sum(np.abs(res-yte))
    #if prnt : print("DTW-KNN: \n",tot,tr)
    return(loss/testsize)

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
    shape = 5

    l = np.zeros(n-1)
    ldtw = np.zeros(n-1)
    lt = np.zeros(n - 1)
    ldtwt = np.zeros(n - 1)


    xtrshape = xtr.shape
    xtr = np.reshape(xtr, (xtrshape[0], xtrshape[1]))

    xteshape = xte.shape
    xte = np.reshape(xte, (xteshape[0], xtrshape[1]))

    print(xte.shape,xtr.shape,yte.shape,ytr.shape)
    with torch.no_grad():
        hiddens_train = np.array(hidden(xtr[:, start:end],model))
        hiddens_test  = np.array(hidden(xte[:, start:end],model))
        tabc = np.exp(1 - tab_dist(hiddens_test, hiddens_train, sp.distance.cosine))

        #On reprend la partie Normale:
        for _ in range(rep):


            hiddens_train, ytr,shuff = shuffle_in_unison(hiddens_train,ytr,True)
            tabc = shuffleaccord(tabc,shuff,1)

            csum = np.cumsum(tabc, axis = 1)
            #print(tabc.shape, tab.shape, l)
            for j in range(1,n):
                i = pas*j
                #print(i,j)
                #print(np.bincount(ytr[:i]))
                l[j-1]    = FSL(tabc[:,:i],csum[:,i], ytr, yte, shape)
                ldtw[j-1] = testknn(xtr[:i],ytr[:i],xte,yte,shape)

            lt += l
            ldtwt += ldtw

            plt.figure()
            sns.set(style = "darkgrid")
            plt.plot(np.arange(pas, pas * n, pas), l, label = "Attention kernel method")
            plt.plot(np.arange(pas, pas * n, pas), ldtw, label = "KNN with" + metric)
            plt.legend(loc = 'best')
            plt.xlabel("Number of data")
            plt.ylabel("Percentage of wrong classifications")
            plt.title("0-1 Loss in percentage for Classification")
            plt.show()

        lt /= (rep)
        ldtwt /= (rep)


    ft = time() - st
    print("Fin en {} min et {} sec".format(ft // 60, ft % 60))

    if show:
        # Partie ou on dessine en 2d les Hidden
        elems = separate(TSNE(n_components = 2).fit_transform(hiddens_train), ytr, 5)
        plt.figure()
        for i, coordi in enumerate(elems):
            #coordi = np.array(coordi)
            print(coordi.shape)
            plt.scatter(coordi[:, 0], coordi[:, 1], label = "{}-th Label".format(i))

        plt.figure()
        sns.set(style = "darkgrid")
        plt.plot(np.arange(pas,pas*n,pas), lt, label = "Attention kernel method")
        plt.plot(np.arange(pas,pas*n,pas), ldtwt, label = "KNN with" + metric)
        plt.legend(loc = 'best')
        plt.xlabel("Number of data")
        plt.ylabel("Percentage of wrong classifications")
        plt.title("0-1 Loss in percentage for Classification")
        plt.show()

        plt.legend(loc = 'best')
        plt.show()


    return(lt[-1],model)


def swigitty(lng,pas):

    classacc = np.zeros(lng-1)
    errtab = np.zeros((lng-1,4500))
    pas_main = 5
    moyen = 1
    xtr, xte,ytr,yte = xsp,X_test,ysp,y_test
    seql = len(xtr)//pas_main
    for j in range(1,lng):
        print(j)
        i = j*pas
        hidden_sizes[0] = i
        #res,model = main(seql,pas_main,xtr,xte,ytr,yte,rep = moyen, show=False)
        model = RNNPred(input_size, hidden_sizes, output_size, output_channels)
        #classacc[j] = res
        errtab[j-1] += model.test_tab(data_e)

    #sns.set(style = "darkgrid")
    #sns.boxplot(np.arange(pas,lng*pas,pas),errtab)
    return(errtab)
    "FAUT QUE JE TROUVE COMMENT plot CECI"


xsp,ysp = shuffle_in_unison(xsp,ysp)


ret = main(31,10,xsp,X_test,ysp,y_test,10, True)
#res = swigitty(10,10)


'''
Retester sur la base de données sans rien changer. (Doing it now)
DTW ? seems to not be working properly ----> Really upsetting.

Finir la fonction Swigitty
- Que regarder par rapport aux résultats de main (pour comparer par rapport à H ?)
- Je ne sais pas comment faire fonctionner boxplot yet mais ça devrait marcher la je crois ???

Améliorer les performances de main car avec le DTW ça ne marche pas très vite
Pour améliorer la fonction main : 
- Essayer une version numpy pour shuffle les éléments, j'ai l'impression que cela bouffe tout le temps, maybe a numba version





Rajouter des commentaires, éventuellement ...

'''
