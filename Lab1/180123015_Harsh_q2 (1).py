import matplotlib.pyplot as plt
import math
import numpy as np

T=5
sigma=0.3
r=0.05
k=105
S_0=100

def binomial_call(S, K, T, r, u, d, M):
    delta = T/M
    p = (math.exp(r*delta)-d)/(u-d)
    C = {}
    if(d >= math.exp(r*delta) or u <= math.exp(r*delta)):
        print("Arbitrage for M:", M)
    for m in range(0, M+1):
            C[(M, m)] = max(0,S*(u**(m))*(d**(M-m))-K)
    for k in range(M-1, -1, -1):
        for m in range(0,k+1):
            C[(k, m)] = (math.exp(-r*delta))*(p*C[(k+1,m+1)]+(1-p)*C[(k+1,m)])
    return C

def binomial_put(S, K, T, r, u, d, M):
    delta = T/M
    p = (math.exp(r*delta)-d)/(u-d)
    P = {}
    if(d >= math.exp(r*delta) or u <= math.exp(r*delta)):
        print("Arbitrage for M:", M)
    for m in range(0, M+1):
            P[(M, m)] = max(0,K-S*(u**(m))*(d**(M-m)))
    for k in range(M-1, -1, -1):
        for m in range(0,k+1):
            P[(k, m)] = (math.exp(-r*delta))*(p*P[(k+1,m+1)]+(1-p)*P[(k+1,m)])
    return P

for i in range(0,2):
    call_prices = []
    put_prices = []
    x = []
    if(i==0):
        for M in range(1,201):
            
            delta=T/M
            u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
            d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
            call= binomial_call(S_0, k, T, r, u, d, M)
            put= binomial_put(S_0, k, T, r, u, d, M)
            call_prices.append(call[0,0])
            put_prices.append(put[0,0])
            x = np.linspace(0,200,200)
        plt.plot(x, call_prices, label = "Call Option price")
        plt.plot(x, put_prices, label = "Put Option price")
        plt.title('Option Prices v/s M for step increment value = 1')

    else:
        for M in range(1,201,5):
            
            delta=T/M
            u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
            d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
            call= binomial_call(S_0, k, T, r, u, d, M)
            put= binomial_put(S_0, k, T, r, u, d, M)
            call_prices.append(call[0,0])
            put_prices.append(put[0,0])
            x = np.linspace(0,200,40)
        plt.plot(x, call_prices, label = "Call Option price")
        plt.plot(x, put_prices, label = "Put Option price")
        plt.title('Option Prices v/s M for step increment value = 5')
    
    plt.legend()
    plt.show()
    plt.clf()    
