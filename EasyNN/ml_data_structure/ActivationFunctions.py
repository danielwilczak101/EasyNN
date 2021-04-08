import math as ma


def id_fxn(x):

    return x
#end id_fxn

def relu_fxn(x):

    return max(0, x)
#end relu_fxn

def leaky_relu_fxn(x):

    leak = 0.01
    return max(leak*x, x)
#end leaky_relu_fxn

def para_relu_fxn(x, leak):

    return max(leak*x, x)
#end para_relu_fxn

def elu_fxn(x):

    scalar = 1

    if max(0, x) == x:
        return x

    else:
        return scalar*(ma.e**x - 1)

#end elu_fxn

def gelu_fxn(x):

    p1 = (2/ma.pi)**0.5
    p2 = x + 0.044715*x**3
    return 0.5*x*(1 + ma.tanh(p1*p2))
#end gelu_fxn

def softplus_fxn(x):

    return ma.log(1 + ma.e**x)
#end softplus_fxn

def logistic_fxn(x):

    return 1 / (1 + ma.e**-x)
#end logistic_fxn

def tanh_fxn(x):

    numerator   = ma.e**x - ma.e**-x
    denominator = ma.e**x + ma.e**-x
    return numerator/denominator
#end tanh_fxn

def arctan_fxn(x):

    return ma.atan(x)
#end arctan_fxn




def main():

    x = 3
    print(gelu_fxn(x))

main()