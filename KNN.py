from RNN import *
import scipy.spatial as sp
import tslearn.metrics as tsmetrics
import tslearn.neighbors as tsn
from sklearn.neighbors import KNeighborsClassifier

def distance(a,b):
    return tsmetrics.dtw(a,b)

def euclideandist(a,b):
    return(np.linalg.norm(a-b))

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
    return(torch.tensor(x.reshape((1,1,seq_len))).float())


@jit
def FSL(tab,csum,lb,true_lb,shape, prnt = False):
    '''
    FEW SHOT LEARNING -PEW PEW, You're learned-
    lb : labels of train
    true_lb : label of test
    '''
    loss = 0
    tot = np.zeros(shape)
    nub = 0
    for e,ligne in enumerate(tab):
        a = np.zeros(shape)
        for i,x in enumerate(ligne):
            a[lb[i]-1] += x
        #a /= csum[e]
        #a /= tr
        res = np.argmax(a)
        tot[res] += 1
        loss += (res+1 != true_lb[e])
    if prnt : print("FSL :\n",tot,tr)
    return(loss/testsize)


'''def testknn(xtr,ytr,xte,yte,shap,prnt = False):
    #tot = np.zeros(shap)
    #tr = np.zeros(shap)
    clf = KNeighborsClassifier(n_neighbors = 1)
    res = clf.fit(xtr,ytr).predict(xte)
    loss = np.sum((res!=yte))
    #if prnt : print("DTW-KNN: \n",tot,tr)
    return(loss/testsize)'''

@jit
def testknn(tab,lab_train,i,k,true_lab,shape,prnt = False):
    '''
    tab tableau des distances, ici test en lignes, train en colonnes.
    '''
    loss = 0
    tot = np.zeros(shape)
    for e,ligne in enumerate(tab):
        a = np.zeros(shape)
        idx = np.argsort(ligne)[:min(i,k)]
        labels = np.take(lab_train,idx)
        for x in labels:
            a[x-1] +=1
        res = np.argmax(a)
        tot[res] += 1
        loss += (res +1 != true_lab[e])
    if prnt : print("KNN: \n",tot,tr)
    return(loss/testsize)


@jit
def hidden(ax,model):
    return [(model.forward(transform(x), True)).numpy() for x in ax]


def main(n,pas,xtr,xte,ytr,yte,rep = 1, show=False):
    '''
        Moyen plus random pour choisir data/lb
        sauter un peu pour le nbr d'elements de x_train
    '''
    # Initialisation du RNN + entrainement :
    model = code()
    model1 = codeclass()
    model2 = codedbheadclass()
    model3 = codedbheadpred()
    st = time()
    shape = 5

    l = np.zeros(n-1)
    ldtw = np.zeros(n-1)
    l1 = np.zeros(n-1)
    l2 = np.zeros(n-1)
    l3 = np.zeros(n-1)

    lt = np.zeros(n - 1)
    lt1 = np.zeros(n - 1)
    lt2 = np.zeros(n - 1)
    lt3 = np.zeros(n - 1)
    ldtwt = np.zeros(n - 1)

    # Ajout des bases de données d'apprentissage :
        # De test
    au1 = xte[:, end:end + length, :]
    au2 = (yte - 1).reshape(yte.shape[0], 1, 1)
    au = np.append(au1, au2, axis = 1)

    data_te = Dataset(xte[:, start:end, :], yte-1)
    data_e_db = torch.utils.data.DataLoader(data_te, batch_size = 24, shuffle = False)

    data_te = Dataset(xte, yte - 1)
    data_e_class = torch.utils.data.DataLoader(data_te, batch_size = 24, shuffle = False)

        # De Train
    au1 = xtr[:, end:end + length, :]
    au2 = (ytr - 1).reshape(ytr.shape[0], 1, 1)
    au = np.append(au1, au2, axis = 1)
    optimizer = optim.SGD(model.parameters(), lr = 0.03)


    xtr1 = np.copy(xtr)
    xte1 = np.copy(xte[:1500])

    xtr = xtr[:,:,0]
    xte = xte[:1500,:,0]
    k=1


    print(xte.shape,xtr.shape,yte.shape,ytr.shape)

    with torch.no_grad():
        ########################## NN Predict ##########################
        hiddens_train = np.array(hidden(xtr[:, start:end],model))
        hiddens_test  = np.array(hidden(xte[:, start:end],model))


        ########################## DB Predict ##########################
        hiddens_train3 = np.array(hidden(xtr[:, start:end],model3))
        hiddens_test3  = np.array(hidden(xte[:, start:end],model3))

        tabc = np.exp(1 - tab_dist(hiddens_test, hiddens_train, sp.distance.cosine))
        tabc3 = np.exp(1 - tab_dist(hiddens_test3, hiddens_train3, sp.distance.cosine))

    tab = tab_dist(xte,xtr,euclideandist)
    #On reprend la partie Normale:
    for _ in range(rep):
        if rep != 1:
            hiddens_train, ytr,shuff = shuffle_in_unison(hiddens_train,ytr,True)
            tabc = shuffleaccord(tabc,shuff,1)

            tabc3 = shuffleaccord(tabc3,shuff,1)

            tab  = shuffleaccord(tab,shuff,1)
            xtr  = shuffleaccord(xtr,shuff,0)

        csum = np.cumsum(tabc, axis = 1)
        csum3 = np.cumsum(tabc3, axis = 1)


        #print(tabc.shape, tab.shape, l)
        for j in range(1,n):
            timee = time()
            print(j)
            #################### Elements pour la classification #################
            data_tr = Dataset(xtr1[(j-1)*pas:j*pas], ytr[(j-1)*pas:j*pas] - 1)
            data_r_class = torch.utils.data.DataLoader(data_tr, batch_size = pas, shuffle = False)
            data_tr = Dataset(xtr1[(j-1)*pas:j*pas, start:end, :], au[(j-1)*pas:j*pas])
            data_r_db = torch.utils.data.DataLoader(data_tr, batch_size = pas, shuffle = True)

            print(np.bincount(ytr[(j-1)*pas:j*pas]))


            #################### Fitting ##################
            model1.fit(data_r_class, 50, optimizer)
            model2.fit(data_r_db,50,optimizer)


            i = pas*j
            #print(i,j)
            l[j-1]  = FSL(tabc[:,:i],csum[:,i], ytr, yte, shape, False)
            l1[j-1] = model1.test_acc(data_e_class,0.1)
            l2[j-1] = model2.test_acc(data_e_db,0.1,True)
            l3[j-1] = FSL(tabc3[:,:i],csum3[:,i], ytr, yte, shape, False)

            #l[j-1] = testknn(tabc[:,:i],ytr[:i],i,5,yte,shape)
            ldtw[j-1] = testknn(tab[:,:i],ytr[:i],i,k,yte,shape)
            print(l[j-1],l1[j-1],l2[j-1],l3[j-1],ldtw[j-1])



        lt += l
        lt1 += l1
        lt2 += l2
        lt3 += l3
        ldtwt += ldtw

        plt.figure()
        sns.set(style = "darkgrid")
        plt.plot(np.arange(pas, pas * n, pas), lt , label = "Kernel method for prediction based model")
        plt.plot(np.arange(pas, pas * n, pas), lt1, label = "Classification based NN")
        plt.plot(np.arange(pas, pas * n, pas), lt2, label = "Classification based Double head NN ")
        plt.plot(np.arange(pas, pas * n, pas), lt3, label = "Kernel method for Double head prediction based model")
        plt.plot(np.arange(pas, pas * n, pas), ldtwt, label = "{}-NN with Euclidian ".format(k) + metric)
        plt.legend(loc = 'best')
        plt.xlabel("Number of data")
        plt.ylabel("Percentage of wrong classifications")
        plt.title("0-1 Loss in percentage for Classification")
        plt.ylim(0, 1)
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

        plt.legend(loc = 'best')
        plt.show()


    return(model, l,l1,l2,l3,ldtw)

'''
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

    sns.set(style = "darkgrid")
    #sns.boxplot(np.arange(pas,lng*pas,pas),errtab)
    return(errtab)
    "FAUT QUE JE TROUVE COMMENT plot CECI"

'''



xsp,ysp = shuffle_in_unison(xsp,ysp)

tr = np.bincount(y_train)[1:]/len(y_train)
hidden_sizes[0] = 64
ret = main(25,15,xsp,X_test,ysp,y_test,1)
#res = swigitty(10,10)


'''
Retester sur la base de données sans rien changer. (Doing it now)
DTW ? seems to not be working properly ----> Really upsetting.

Finir la fonction Swigitty
- Que regarder par rapport aux résultats de main (pour comparer par rapport à H ?)
- Je ne sais pas comment faire fonctionner boxplot yet mais ça devrait marcher la je crois ???

KNN :
- Améliorer le code KNN pour ne pas avoir a recalculer les distances (Nécéssite ma propre fonction KNN)
- Utiliser DTW ?

Seaborn error bar ----> finir swigitty.

--------------------------------

Classes minoritaires .
Matrice de "confusion".




FAIRE UNE SAUVEGARDE AUTOMATIQUE DES PLOTS dans des dossiers, ce serait bien.

Rajouter des commentaires, éventuellement ...


---------------------
différents tests :

Différents h --> ne change rien cette fois. (wierd)*
j'aurai espéré qu'augmenter la diension aurait donné plus de précision niveau des classes peu représentés (malédiction de la grande dimension)
Prob de répartition des classes -> 
-diviser par le nb d'elems dans chaque classe ?
- cette méthode a une eeur importante avec h grand
- 1NN au lieu de kernel ?

'''
