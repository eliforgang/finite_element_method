def bandLU(K,p,q): #LU factorization of a banded matrix
    import numpy as np
    n=len(K)
    for m in range(0,n-1):
        for i in range(m+1,min(m+p+1,n)):
            K[i,m]=K[i,m]/K[m,m]
        for i in range(m+1,min(m+p+1,n)):
            for j in range(m+1,min(m+q+1,n)):
                K[i,j]=K[i,j]-(K[i,m]*K[m,j])
    return K

#use bandforward and bandbackward to solve LUx=A where A is a matrix and
def bandforward(L,f,p): 
    import numpy as np
    n=len(f)
    z=np.zeros(n)
    for i in range(0,n):
        z[i]=f[i]/L[i,i]
        for j in range(max(0,i-p),i):
            z[i]-=L[i,j]*z[j]/L[i,i]
    return z

def bandbackward(U,f,q):
    import numpy as np
    n=len(f)
    w=np.zeros(n)
    for i in range(n-1,-1,-1):
        w[i]=f[i]/U[i,i]
        for j in range(i+1,min(n,i+q+1)):
            w[i]-=U[i,j]*w[j]/U[i,i]
    return w