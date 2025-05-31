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


def quad2dTs(g, noOfIntegPt): #2D standard gaussian quadrature with 4, 6, or 7 integration points
    if noOfIntegPt==4:
        val = (-9/32)*g(1/3,1/3) + (25/96)*g(3/5,1/5) + (25/96)*g(1/5,1/5) + (25/96)*g(1/5,3/5)
    elif noOfIntegPt==6:
        w = np.array([137/2492,137/2492,137/2492,1049/9392,1049/9392,1049/9392])
        xi = np.array([1280/1567,287/3134,287/3134,575/5319,2372/5319,2372/5319])
        eta = np.array([287/3134,1280/1567,287/3134,2372/5319,575/5319,2372/5319])
        val = 0
        for i in range(0,6):
            val+=w[i]*g(xi[i],eta[i])
    elif noOfIntegPt==7:
        w = np.array([9/80,352/5590,352/5590,352/5590,1748/26406,1748/26406,1748/26406])
        xi = np.array([1/3,248/311,496/4897,496/4897,248/4153,496/1055,496/1055])
        eta = np.array([1/3,496/4897,248/311,496/4897,496/1055,248/4153,496/1055])
        val = 0
        for i in range(0,7):
            val+=w[i]*g(xi[i],eta[i])
    else:
        val = 0
    return val

def quad2dTri(f, x1, y1, x2, y2, x3, y3, noOfIntegPt): #2D general gaussian quadrature with 4, 6, or 7 integration points
    a11 = x2 - x1
    a21 = y2 - y1
    a12 = x3 - x1
    a22 = y3 - y1
    det = (a11*a22 - a12*a21)
    jacobian = abs(det)
    g = lambda xi, eta: f(x1 + (x2 - x1)*xi + (x3 -x1)*eta,y1 + (y2 - y1)*xi + (y3 - y1)*eta)
    val = jacobian*quad2dTs(g, noOfIntegPt)
    return val