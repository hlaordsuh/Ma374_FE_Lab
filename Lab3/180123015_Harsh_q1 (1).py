import math
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time



T=1
sigma=0.2
r=0.08
S_0=100

def isKthBitSet(n, k): 
    if n & (1 << (k - 1)): 
        return True
    else: 
        return False 

def lookback_price(S0,T,m,r,sigma):
    delta = T/m
    u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
    d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))

    p = (math.exp(r*delta)-d)/(u-d)
    lookback_option_price = 0
    for k in range(0,2**m):
        price = []
        price.append(S0)
        cnt = 0
        for i in range(1,m+1):
            xx = 0
            if isKthBitSet(k,i):
                cnt+=1
                xx = price[-1]*u
            else:
                xx = price[-1]*d
            price.append(xx)
        Smax = np.max(price)
        lookback_payoff = Smax-price[-1]
        lookback_option_price += math.pow(p,cnt) * math.pow((1-p),m-cnt) * lookback_payoff
        
    lookback_option_price = lookback_option_price/math.exp(r*T) 
    return lookback_option_price

def lookback_price_t(S0,T,m,r,sigma,t):
    
    print("For t =",t,":")
    print(" LOOKBACK OPTION PRICE")
    delta = T/m
    u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
    d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))

    p = (math.exp(r*delta)-d)/(u-d)
    
    
    t_rem = (int) (round((m - t/delta))) #number of time periods remaining
    t_occ = (int) (round((t/delta))) #number of time periods occurred
    
    for l in range(0,2**t_occ):
        pr = []
        pr.append(S0)
        cn = 0
        for j in range(1,t_occ+1):
            xx = 0
            if isKthBitSet(l,j):
                cn+=1
                xx = pr[-1]*u
            else:
                xx = pr[-1]*d
            pr.append(xx)
        
        S_init = S0 * math.pow(u,cn) * math.pow(d,t_occ-cn)
        S_max = np.max(pr)
        lookback_option_price = 0
        for k in range(0,2**t_rem):
            price = []
            price.append(S_init)
            cnt = 0
            for i in range(1,t_rem+1):
                xx = 0
                if isKthBitSet(k,i):
                    cnt+=1
                    xx = price[-1]*u
                else:
                    xx = price[-1]*d
                price.append(xx)
            
            Smax = np.max(price)
            Smax = max(Smax,S_max)
            lookback_payoff = Smax-price[-1]
            lookback_option_price += math.pow(p,cnt) * math.pow((1-p),t_rem-cnt) * lookback_payoff
        
        print("{:.6f}".format(lookback_option_price/math.exp(r*delta*t_rem)))


print("q1(a)")
t1 = time.time()
print("m =",5,",lookback option price =",lookback_price(100,1,5,0.08,0.2))
t = time.time()
print("for m= 5, Computational time = ",t-t1)
t1 = time.time()
print("m =",10,",lookback option price =",lookback_price(100,1,10,0.08,0.2))
t = time.time()
print("for m= 10, Computational time = ",t-t1)
t1 = time.time()
print("m =",15,",lookback option price =",lookback_price(100,1,15,0.08,0.2))
t = time.time()
print("for m= 15, Computational time = ",t-t1)
t1 = time.time()
print("m =",20,",lookback option price =",lookback_price(100,1,20,0.08,0.2))
t = time.time()
print("for m= 20, Computational time = ",t-t1)

print("")
print("q1(b)")
xx = []
yy = []

for i in range(5,21):
    xx.append(i)
    yy.append(lookback_price(100,1,i,0.08,0.2))
    
plt.scatter(xx, yy,c='r',s=10)
plt.plot(xx, yy)
plt.xlabel("Value of M")
plt.ylabel("Initial Option Price")
plt.xticks(xx)
plt.title("Lookback Option Price at t=0 vs M")
plt.grid(True)
plt.show()

print("")
print("q1(c)")
print("")
lookback_price_t(100,1,5,0.08,0.2,0)
lookback_price_t(100,1,5,0.08,0.2,0.2)
lookback_price_t(100,1,5,0.08,0.2,0.4)
lookback_price_t(100,1,5,0.08,0.2,0.6)
lookback_price_t(100,1,5,0.08,0.2,0.8)
lookback_price_t(100,1,5,0.08,0.2,1)


