#M, K, and F for solving MU* + KU = F for U (intial boundary value problem)

def meij(e, i, j, xh, shapeFn, noOfIntegPt): 
    #eth element, ith and jth shape function, xh is a vector, shapeFn = 1 or 2, noOfIntegPt is how many integration points
    x1 = xh[e-1]
    x2 = xh[e]
    psi_i = lambda x: shapeFn1d(i,x,x1,x2,shapeFn)
    psi_j = lambda x: shapeFn1d(j,x,x1,x2,shapeFn)
    fn = lambda x: psi_i(x)*psi_j(x)
    y = gaussQuad1d(fn,x1,x2,noOfIntegPt) #integrate
    return y


def keij(a, c, e, i, j, xh, shapeFn, noOfIntegPt): #a(x), c(x) functions
    x1 = xh[e-1]
    x2 = xh[e]
    psi_i = lambda x: shapeFn1d(i,x,x1,x2,shapeFn)
    psi_j = lambda x: shapeFn1d(j,x,x1,x2,shapeFn)
    psi_i_der = lambda x: shapeFnDer1d(i,x,x1,x2,shapeFn)
    psi_j_der = lambda x: shapeFnDer1d(j,x,x1,x2,shapeFn)
    fn_c = lambda x: c(x)*psi_i(x)*psi_j(x)
    fn_a = lambda x: a(x)*psi_i_der(x)*psi_j_der(x)
    fn = lambda x: fn_c(x) + fn_a(x)
    y = gaussQuad1d(fn,x1,x2,noOfIntegPt) #integrate
    return y


def fei(a, c, f, p0, e, i, xh, shapeFn, noOfIntegPt): #f(x) function, p0 Dirichlet boundary condition at x=0
    x1 = xh[e-1]
    x2 = xh[e]
    psi_i = lambda x: shapeFn1d(i,x,x1,x2,shapeFn)
    psi_i_der = lambda x: shapeFnDer1d(i,x,x1,x2,shapeFn)
    psi_1_1 = lambda x: shapeFn1d(1,x,xh[0],xh[1],shapeFn)
    psi_1_1_der = lambda x: shapeFnDer1d(1,x,xh[0],xh[1],shapeFn)
    if e==1:
        G = lambda x: p0*psi_1_1(x)
        G_der = lambda x: p0*psi_1_1_der(x)
    else:
        G = lambda x: 0
        G_der = lambda x: 0
    fn = lambda x: (f(x)*psi_i(x)) - (a(x)*G_der(x)*psi_i_der(x)) - (c(x)*G(x)*psi_i(x))
    y = gaussQuad1d(fn,x1,x2,noOfIntegPt)
    return y


def massM(xh, shapeFn, noOfIntegPt): #consistent mass matrix M
    import numpy as np
    from scipy.sparse import csr_matrix
    row = []
    col = []
    data = []
    if shapeFn==1:
        data.append(meij(1, 2, 2, xh, shapeFn, noOfIntegPt))
        row.append(0)
        col.append(0)
        for e in range(2,len(xh)):
            data.append(meij(e, 1, 1, xh, shapeFn, noOfIntegPt)) #top left
            row.append(e-2)
            col.append(e-2)
            data.append(meij(e, 2, 2, xh, shapeFn, noOfIntegPt)) #bottom right
            row.append(e-1)
            col.append(e-1)
            data.append(meij(e, 1, 2, xh, shapeFn, noOfIntegPt)) #top right
            row.append(e-2)
            col.append(e-1)
            data.append(meij(e, 2, 1, xh, shapeFn, noOfIntegPt)) #bottom left
            row.append(e-1)
            col.append(e-2)
        M=csr_matrix((data, (row, col)), shape=(len(xh)-1,len(xh)-1)).toarray()
    elif shapeFn==2:
        data.append(meij(1, 2, 2, xh, shapeFn, noOfIntegPt)) #top left
        row.append(0)
        col.append(0)
        data.append(meij(1, 2, 3, xh, shapeFn, noOfIntegPt)) #top right
        row.append(0)
        col.append(1)
        data.append(meij(1, 3, 2, xh, shapeFn, noOfIntegPt)) #bottom left
        row.append(1)
        col.append(0)
        data.append(meij(1, 3, 3, xh, shapeFn, noOfIntegPt)) #bottom right
        row.append(1)
        col.append(1)
        for e in range(2,len(xh)):
            data.append(meij(e, 1, 1, xh, shapeFn, noOfIntegPt)) #top left
            row.append(2*(e-1)-1)
            col.append(2*(e-1)-1)
            data.append(meij(e, 1, 2, xh, shapeFn, noOfIntegPt)) #top middle
            row.append(2*(e-1)-1)
            col.append(2*e-2)
            data.append(meij(e, 1, 3, xh, shapeFn, noOfIntegPt)) #top right
            row.append(2*(e-1)-1)
            col.append(2*e-1)
            data.append(meij(e, 2, 1, xh, shapeFn, noOfIntegPt)) #middle left
            row.append(2*e-2)
            col.append(2*(e-1)-1)
            data.append(meij(e, 2, 2, xh, shapeFn, noOfIntegPt)) #middle
            row.append(2*e-2)
            col.append(2*e-2)
            data.append(meij(e, 2, 3, xh, shapeFn, noOfIntegPt)) #middle right
            row.append(2*e-2)
            col.append(2*e-1)
            data.append(meij(e, 3, 1, xh, shapeFn, noOfIntegPt)) #bottom left
            row.append(2*e-1)
            col.append(2*(e-1)-1)
            data.append(meij(e, 3, 2, xh, shapeFn, noOfIntegPt)) #bottom middle
            row.append(2*e-1)
            col.append(2*e-2)
            data.append(meij(e, 3, 3, xh, shapeFn, noOfIntegPt)) #bottom right
            row.append(2*e-1)
            col.append(2*e-1)
        M=csr_matrix((data, (row, col)), shape=(2*len(xh)-2,2*len(xh)-2)).toarray()
    else:
        M=[]
    return M


def stiffK(a, c, xh, shapeFn, noOfIntegPt): #stiff matrix K
    import numpy as np
    from scipy.sparse import csr_matrix
    row = []
    col = []
    data = []
    if shapeFn==1:
        data.append(keij(a, c, 1, 2, 2, xh, shapeFn, noOfIntegPt))
        row.append(0)
        col.append(0)
        for e in range(2,len(xh)):
            data.append(keij(a, c, e, 1, 1, xh, shapeFn, noOfIntegPt)) #top left
            row.append(e-2)
            col.append(e-2)
            data.append(keij(a, c, e, 2, 2, xh, shapeFn, noOfIntegPt)) #bottom right
            row.append(e-1)
            col.append(e-1)
            data.append(keij(a, c, e, 1, 2, xh, shapeFn, noOfIntegPt)) #top right
            row.append(e-2)
            col.append(e-1)
            data.append(keij(a, c, e, 2, 1, xh, shapeFn, noOfIntegPt)) #bottom left
            row.append(e-1)
            col.append(e-2)
        K=csr_matrix((data, (row, col)), shape=(len(xh)-1,len(xh)-1)).toarray()
    elif shapeFn==2:
        data.append(keij(a, c, 1, 2, 2, xh, shapeFn, noOfIntegPt)) #top left
        row.append(0)
        col.append(0)
        data.append(keij(a, c, 1, 2, 3, xh, shapeFn, noOfIntegPt)) #top right
        row.append(0)
        col.append(1)
        data.append(keij(a, c, 1, 3, 2, xh, shapeFn, noOfIntegPt)) #bottom left
        row.append(1)
        col.append(0)
        data.append(keij(a, c, 1, 3, 3, xh, shapeFn, noOfIntegPt)) #bottom right
        row.append(1)
        col.append(1)
        for e in range(2,len(xh)):
            data.append(keij(a, c, e, 1, 1, xh, shapeFn, noOfIntegPt)) #top left
            row.append(2*(e-1)-1)
            col.append(2*(e-1)-1)
            data.append(keij(a, c, e, 1, 2, xh, shapeFn, noOfIntegPt)) #top middle
            row.append(2*(e-1)-1)
            col.append(2*e-2)
            data.append(keij(a, c, e, 1, 3, xh, shapeFn, noOfIntegPt)) #top right
            row.append(2*(e-1)-1)
            col.append(2*e-1)
            data.append(keij(a, c, e, 2, 1, xh, shapeFn, noOfIntegPt)) #middle left
            row.append(2*e-2)
            col.append(2*(e-1)-1)
            data.append(keij(a, c, e, 2, 2, xh, shapeFn, noOfIntegPt)) #middle
            row.append(2*e-2)
            col.append(2*e-2)
            data.append(keij(a, c, e, 2, 3, xh, shapeFn, noOfIntegPt)) #middle right
            row.append(2*e-2)
            col.append(2*e-1)
            data.append(keij(a, c, e, 3, 1, xh, shapeFn, noOfIntegPt)) #bottom left
            row.append(2*e-1)
            col.append(2*(e-1)-1)
            data.append(keij(a, c, e, 3, 2, xh, shapeFn, noOfIntegPt)) #bottom middle
            row.append(2*e-1)
            col.append(2*e-2)
            data.append(keij(a, c, e, 3, 3, xh, shapeFn, noOfIntegPt)) #bottom right
            row.append(2*e-1)
            col.append(2*e-1)
        K=csr_matrix((data, (row, col)), shape=(2*len(xh)-2,2*len(xh)-2)).toarray()
    else:
        K=[]
    return K


def loadF(a, c, f, p0, QL, xh, shapeFn, noOfIntegPt): #load vector F; QL = a(du/dn)|x=L
    import numpy as np
    if shapeFn==1:
        F=np.zeros(len(xh)-1)
        F[0]=fei(a, c, f, p0, 1, 2, xh, shapeFn, noOfIntegPt)
        for e in range(2,len(xh)):
            F[e-2]+=fei(a, c, f, p0, e, 1, xh, shapeFn, noOfIntegPt)
            F[e-1]+=fei(a, c, f, p0, e, 2, xh, shapeFn, noOfIntegPt)
        F[-1]+=QL
    elif shapeFn==2:
        F=np.zeros(2*(len(xh)-1))
        F[0]=fei(a, c, f, p0, 1, 2, xh, shapeFn, noOfIntegPt)
        F[1]=fei(a, c, f, p0, 1, 3, xh, shapeFn, noOfIntegPt)
        for e in range(2,len(xh)):
            F[2*(e-1)-1]+=fei(a, c, f, p0, e, 1, xh, shapeFn, noOfIntegPt)
            F[2*(e-1)]+=fei(a, c, f, p0, e, 2, xh, shapeFn, noOfIntegPt)
            F[2*(e-1)+1]+=fei(a, c, f, p0, e, 3, xh, shapeFn, noOfIntegPt)
        F[-1]+=QL
    return F


def L2norm1d(f,a,b,noOfEle): #1D L2 norm
    import numpy as np
    x=np.zeros(noOfEle+1)
    x[0]=a
    for i in range(1,noOfEle+1):
        x[i]=x[i-1] + ((b-a)/noOfEle)
    fn = lambda x: f(x)**2
    sum_elements = 0
    for j in range(0,noOfEle):
        sum_elements += gaussQuad1d(fn,x[j],x[j+1],3)
    y=np.sqrt(sum_elements)
    return y

################################################################
def shapeFn1d(i,x,x1,x2,p): #1D shape functions
    if x1<=x and x<=x2:
        h=x2-x1
        if p==1:
            if i==1:
                y=(x2-x)/h
            elif i==2:
                y=(x-x1)/h
            else:
                y=0
        elif p==2:
            if i==1:
                y=(x2 - x)*(x2 - 2*x + x1)/(h**2)
            elif i==2:
                y=4*(x - x1)*(x2 - x)/(h**2)
            elif i==3:
                y=(x1 - x)*(x2 - 2*x +x1)/(h**2)
            else:
                y=0
        else:
            y=0
    else:
        y=0
    return y


def shapeFnDer1d(i,x,x1,x2,p): #1D shape function derivatives
    if x1<=x and x<=x2:
        h=x2-x1
        if p==1:
            if i==1:
                y=-1/h
            elif i==2:
                y=1/h
            else:
                y=0
        elif p==2:
            if i==1:
                y=(4*x - 3*x2 - x1)/(h**2)
            elif i==2:
                y=4*(-2*x + x2 + x1)/(h**2)
            elif i==3:
                y=(4*x - 3*x1 - x2)/(h**2)
            else:
                y=0
        else:
            y=0
    else:
        y=0
    return y
    
    
def gaussQuad1d(fn,lowerLimit,upperLimit,noOfIntegPt): #1D general gaussian quadrature up to 5 integration points
    import math
    g = lambda x: fn(upperLimit + ((upperLimit - lowerLimit)/2)*(x-1))
    if noOfIntegPt==2:
        y = ((upperLimit - lowerLimit)/2)*(g(-1/(math.sqrt(3))) + g(1/(math.sqrt(3))))
        return y
    elif noOfIntegPt==3:
        y = ((upperLimit - lowerLimit)/2)*((5/9)*g(-math.sqrt(3/5)) + (8/9)*g(0) + (5/9)*g(math.sqrt(3/5)))
        return y
    elif noOfIntegPt==4:
        y = ((upperLimit - lowerLimit)/2)*(((18-math.sqrt(30))/36)*g(-math.sqrt((3+(2*math.sqrt(6/5)))/7)) + 
                                          ((18+math.sqrt(30))/36)*g(-math.sqrt((3-(2*math.sqrt(6/5)))/7)) + 
                                          ((18+math.sqrt(30))/36)*g(math.sqrt((3-(2*math.sqrt(6/5)))/7)) +
                                          ((18-math.sqrt(30))/36)*g(math.sqrt((3+(2*math.sqrt(6/5)))/7)))
        return y
    elif noOfIntegPt==5:
        y = ((upperLimit - lowerLimit)/2)*(((322-(13*math.sqrt(70)))/900)*g(-(1/3)*math.sqrt(5+2*math.sqrt(10/7))) + 
                                           ((322+(13*math.sqrt(70)))/900)*g(-(1/3)*math.sqrt(5-2*math.sqrt(10/7))) +
                                           (128/225)*g(0) + 
                                           ((322+(13*math.sqrt(70)))/900)*g((1/3)*math.sqrt(5-2*math.sqrt(10/7))) +
                                           ((322-(13*math.sqrt(70)))/900)*g(math.sqrt((1/3)*math.sqrt(5+2*math.sqrt(10/7)))))
        return y
    else:
        print('Number of integers must be two, three, four, or five')