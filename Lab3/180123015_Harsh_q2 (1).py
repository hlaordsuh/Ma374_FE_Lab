import math
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time

T=1
sigma=0.2
r=0.08
S_0=100

mp={}
def helper(n,mx,S,T,r,sigma,m,u,d,p,q,delta):
    if (mx,S) in mp:
        return mp[(mx,S)]

    if(n==m):
        mp[(mx,S)]=mx-S
        return mx-S

    up=helper(n+1,max(mx,S*u),S*u,T,r,sigma,m,u,d,p,q,delta)
    dn=helper(n+1,max(mx,S*d),S*d,T,r,sigma,m,u,d,p,q,delta)
    pc=(math.exp(-r*delta))*(p*up+q*dn)
    mp[(mx,S)]=pc
    return pc

def lookback_price_markov(S0,T,m,r,sigma):
    delta = T/m
    u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
    d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))

    p = (math.exp(r*delta)-d)/(u-d)
    q=1-p
    mp.clear()
    ans=helper(0,S0,S0,T,r,sigma,m,u,d,p,q,delta)
    return ans

print("q2")
t1 = time.time()
print("m =",5,",lookback option price =",lookback_price_markov(100,1,5,0.08,0.2))
t = time.time()
print("for m= 5, Computational time for markov method = ",t-t1)
t1 = time.time()
print("m =",10,",lookback option price =",lookback_price_markov(100,1,10,0.08,0.2))
t = time.time()
print("for m= 10, Computational time for markov method = ",t-t1)
t1 = time.time()
print("m =",25,",lookback option price =",lookback_price_markov(100,1,25,0.08,0.2))
t = time.time()
print("for m= 25, Computational time for markov method = ",t-t1)
t1 = time.time()
print("m =",50,",lookback option price =",lookback_price_markov(100,1,50,0.08,0.2))
t = time.time()
print("for m= 50, Computational time for markov method = ",t-t1)
