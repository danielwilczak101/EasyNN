import math as ma


def id_fxn(x):

    return x
#end id_fxn

def d_id_fxn(x):

    return 1
#end d_id_fxn


def relu_fxn(x):

    return max(0, x)
#end relu_fxn

def d_relu_fxn(x):

    if x > 0:
        return 1
    else:
        return 0
#end d_relu_fxn


def leaky_relu_fxn(x):

    leak = 0.01
    return max(leak*x, x)
#end leaky_relu_fxn

def d_leaky_relu_fxn(x):

    leak = 0.01
    if x > 0:
        return 1
    else:
        return leak
#end d_leaky_relu_fxn


def para_relu_fxn(x, leak):

    return max(leak*x, x)
#end para_relu_fxn

def d_para_relu_fxn(x, leak):

    if x > 0:
        return 1
    else:
        return leak
#end d_para_relu_fxn

def elu_fxn(x):

    scalar = 1

    if max(0, x) == x:
        return x

    else:
        return scalar*(ma.e**x - 1)

#end elu_fxn

def d_elu_fxn(x):

    scalar = 1

    if x > 0:
        return 1
    else:
        return scalar*ma.e**x

#end d_elu_fxn

def gelu_fxn(x):

    p1 = (2/ma.pi)**0.5
    p2 = x + 0.044715*x**3
    return 0.5*x*(1 + ma.tanh(p1*p2))
#end gelu_fxn

def d_gelu_fxn(x):

    n1 = 26829*(2**0.5)*(x**3)
    n2 = 200000*(2**0.5)*x
    n3 = 100000*(ma.pi**0.5)
    n4_1 = (8943*(2**0.5)*(x**3)) / n3
    n4_2 = (2*(2**0.5)*x) / (ma.pi**0.5)
    n4 = ma.e**(n4_1 + n4_2)
    n = (n1 + n2 - n3)*n4 - n3

    d1 = n3
    d2 = n4 + 1
    d = d1*(d2**2)

    return n/d + 1
#end d_gelu_fxn

def softplus_fxn(x):

    return ma.log(1 + ma.e**x)
#end softplus_fxn

def d_softplus_fxn(x):

    n = ma.e**x
    d = n + 1
    return n/d
#end d_softplus_fxn

def logistic_fxn(x):

    return 1 / (1 + ma.e**-x)
#end logistic_fxn

def d_logistic_fxn(x):

    n = 1
    d = 4*(ma.cosh(x/2)**2)
    return n/d
#end d_logistic_fxn

def tanh_fxn(x):

    return ma.tanh(x)
#end tanh_fxn

def d_tanh_fxn(x):

    n = 1
    d = ma.cosh(x)**2
    return n/d
#end d_tanh_fxn

def arctan_fxn(x):

    return ma.atan(x)
#end arctan_fxn

def d_arctan_fxn(x):

    n = 1
    d = x**2 + 1
    return n/d
#end d_arctan_fxn


def main():

    x = 2.5
    print(d_gelu_fxn(x))

main()
