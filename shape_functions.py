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
    

def shapeFn2dTs(i, xi, eta, p): #standard triangle 2D shape functions
    if (xi<-1e-12) or (eta<-1e-12) or ((xi + eta)>(1 + 1e-12)):
        z=0
    else:
        if p==1:
            if i==1:
                z = 1 - xi - eta
            elif i==2:
                z = xi
            elif i==3:
                z = eta
            else:
                z = 0
        elif p==2:
            if i==1:
                z = (1 - xi - eta)*(1 - 2*xi - 2*eta)
            elif i==2:
                z = xi*(2*xi - 1)
            elif i==3:
                z = eta*(2*eta - 1)
            elif i==4:
                z = 4*xi*(1 - xi - eta)
            elif i==5:
                z = 4*xi*eta
            elif i==6:
                z = 4*eta*(1 - xi - eta)
            else:
                z = 0
        else:
            z = 0
    return z

def shapeFn2d(i, x, y, x1, y1, x2, y2, x3, y3, p): #general 2D shape functions
    a11 = x2 - x1
    a21 = y2 - y1
    a12 = x3 - x1
    a22 = y3 - y1
    det = (a11*a22 - a12*a21)
    Ainv = np.array([[a22/det,-a12/det],[-a21/det,a11/det]])
    vector = np.array([x-x1,y-y1])
    transform = np.matmul(Ainv,vector)
    xi = transform[0]
    eta = transform[1]
    z = shapeFn2dTs(i, xi, eta, p)
    return z

def shapeFnGrad2dTs(i, xi, eta, p): #standard triangle 2D shape function gradients
    if (xi<-1e-12) or (eta<-1e-12) or ((xi + eta)>(1 + 1e-12)):
        psi_xi = 0
        psi_eta = 0
    else:
        if p==1:
            if i==1:
                psi_xi = -1
                psi_eta = -1
            elif i==2:
                psi_xi = 1
                psi_eta = 0
            elif i==3:
                psi_xi = 0
                psi_eta = 1
            else:
                psi_xi = 0
                psi_eta = 0
        elif p==2:
            if i==1:
                psi_xi = 4*xi + 4*eta - 3
                psi_eta = 4*xi + 4*eta - 3
            elif i==2:
                psi_xi = 4*xi - 1
                psi_eta = 0
            elif i==3:
                psi_xi = 0
                psi_eta = 4*eta - 1
            elif i==4:
                psi_xi = 4 - 8*xi - 4*eta
                psi_eta = -4*xi
            elif i==5:
                psi_xi = 4*eta
                psi_eta = 4*xi
            elif i==6:
                psi_xi = -4*eta
                psi_eta = 4 - 4*xi - 8*eta
            else:
                psi_xi = 0
                psi_eta = 0
        else:
            psi_xi = 0
            psi_eta = 0
    return psi_xi, psi_eta

def shapeFnGrad2d(i, x, y, x1, y1, x2, y2, x3, y3, p): #general triangle 2D shape function gradients
    a11 = x2 - x1
    a21 = y2 - y1
    a12 = x3 - x1
    a22 = y3 - y1
    det = (a11*a22 - a12*a21)
    Ainv = np.array([[a22/det,-a12/det],[-a21/det,a11/det]])
    vector = np.array([x-x1,y-y1])
    transform = np.matmul(Ainv,vector)
    xi = transform[0]
    eta = transform[1]
    gradpsi = np.matmul(Ainv.T,shapeFnGrad2dTs(i, xi, eta, p))
    return gradpsi
