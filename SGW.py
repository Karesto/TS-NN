#Computes the Souped Gromov-Wasserstein

import numpy as np


def integrate(x1, x2, y1, y2):
    x = np.append(x1,x2)
    y = np.cumsum(np.append(y1,y2))
    n = len(x)
    s=0
    for i in range(n-1):
        s += y[i]*(x[i+1]-x[i])
    return np.abs(s)


def mincalc(l,c,coef1,coef2):
    m,ind = np.inf, []
    n,m = len(l),len(c)
    x1, y1 = zip(*sorted(zip(l, coef1)))
    x1, y1 = np.array(x1),np.array(y1)
    for i in range(m):
        x2,y2 = zip(*sorted(zip(c[i], coef2)))
        x2, y2 = np.array(x2), np.array(y2)
        res = integrate(x1,x2,y1,-y2)
        if res < m:
            m = res
        if res == 0:
            ind.append(i)
    return(m,ind)

def swg(c1,c2,coef1,coef2):
    s1, m1 = 0 ,[]
    for j in range (len(c1)):
        print(c1,c2,coef1,coef2)

        r1,r2 = mincalc(c1[j],c2,coef1,coef2)
        s1 += coef1[j]*r1
        m1.append(r2)

    s2, m2 = 0 ,[]
    for j in range(len(c2)):
        r1, r2 = mincalc(c2[j], c1, coef2, coef1)
        s2 += coef2[j] * r1
        m2.append(r2)

    return (s1+s2)/2 , m1, m2


c1 = [[0., 1., 2., 2.],
      [1., 0., 2., 2.],
      [2., 2., 0., 2.],
      [2., 2., 2., 0.]]

c2 = [[0., 1., 2., 2., 2.],
      [1., 0., 2., 2., 2.],
      [2., 2., 0., 2., 2.],
      [2., 2., 2., 0., 1.],
      [2., 2., 2., 1., 0.]]

alph1 = 1/6
alph2 = 1/3
coeff1 = [alph1,alph1,alph2,alph2]
coeff2 = [alph1,alph1,alph2,alph1,alph1]

#Test case at the end for different dimention.
#Note that we are supposed to send the matrix of possible permutations for each row/Matrix. 
#More details surely later
