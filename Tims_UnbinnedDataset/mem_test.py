import numpy as np

def foo():
    a=np.random.random(100_000_000)
    b=np.linspace(1,1e6,100_000_000)
#     c=a*b
#     c = 2*a + 3*b
    a*=b
#     return c