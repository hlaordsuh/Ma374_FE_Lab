import math

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

for M in [1,5,10,20,50,100,200,400]:
    delta=T/M
    u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
    d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
    call_price = binomial_call(S_0, k, T, r, u, d, M)
    put_price = binomial_put(S_0, k, T, r, u, d, M)
    print("For M = ",M)
    print("Call option price:",call_price[0,0]," Put option price:",put_price[0,0])
