import math
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time

S0 = 100
T = 1
r = 0.08
sigma = 0.2
K = 100

def recursive_func1(n,K,S_cur,u,d,p,delta,m):
    if n==m:
        return max(S_cur-K,0)
    up_price = recursive_func1(n+1,K,S_cur*u,u,d,p,delta,m)
    down_price = recursive_func1(n+1,K,S_cur*d,u,d,p,delta,m)
    curr_price = p*up_price + (1-p)*down_price
    curr_price *= math.exp(-r*delta)
    return curr_price
    
def european_call(m):
    delta = T/m
    u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
    d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
    p = ((math.exp(r*delta)) - d)/(u-d)
    return recursive_func1(0,K,S0,u,d,p,delta,m)

mp={}
def erecursive_func1(n,K,S_cur,u,d,p,delta,m):
    if (n,S_cur) in mp:
        return mp[(n,S_cur)]
    if n==m:
        mp[(n,S_cur)]=max(S_cur-K,0)
        return max(S_cur-K,0)
    up_price = erecursive_func1(n+1,K,S_cur*u,u,d,p,delta,m)
    down_price = erecursive_func1(n+1,K,S_cur*d,u,d,p,delta,m)
    curr_price = p*up_price + (1-p)*down_price
    curr_price *= math.exp(-r*delta)
    mp[(n,S_cur)]=curr_price
    return curr_price
    
def efficient_european_call(m):
    delta = T/m
    u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
    d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
    p = ((math.exp(r*delta)) - d)/(u-d)
    mp.clear()
    return erecursive_func1(0,K,S0,u,d,p,delta,m)

# # print("q3")
t1 = time.time()
print("m =",5,",european option price =",european_call(5))
t = time.time()
print("for m= 5, Computational time  = ",t-t1)
t1 = time.time()
print("m =",10,",european option price =",european_call(10))
t = time.time()
print("for m= 10, Computational time  = ",t-t1)
t1 = time.time()
print("m =",15,",european option price =",european_call(15))
t = time.time()
print("for m= 15, Computational time  = ",t-t1)
t1 = time.time()
print("m =",20,",european option price =",european_call(20))
t = time.time()
print("for m= 20, Computational time  = ",t-t1)

t1 = time.time()
print("m =",5,",efficient european option price =",efficient_european_call(5))
t = time.time()
print("for m= 5, Computational time  = ",t-t1)
t1 = time.time()
print("m =",10,",efficient european option price =",efficient_european_call(10))
t = time.time()
print("for m= 10, Computational time  = ",t-t1)
t1 = time.time()
print("m =",15,",efficient european option price =",efficient_european_call(15))
t = time.time()
print("for m= 15, Computational time  = ",t-t1)
t1 = time.time()
print("m =",20,",efficient european option price =",efficient_european_call(20))
t = time.time()
print("for m= 20, Computational time  = ",t-t1)
print("m =",25,",efficient european option price =",efficient_european_call(25))
t = time.time()
print("for m= 25, Computational time  = ",t-t1)
t1 = time.time()
print("m =",50,",efficient european option price =",efficient_european_call(50))
t = time.time()
print("for m= 50, Computational time  = ",t-t1)