import matplotlib.pyplot as plt
import math
import numpy as np
from mpl_toolkits import mplot3d 

T=1
sigma=0.2
r=0.08
k=100
S_0=100
M=10

def isKthBitSet(n, k): 
    if n & (1 << (k - 1)): 
        return True
    else: 
        return False 

def binomial_call(S, K, T, r, u, d, M):
    delta = T/M
    p = (math.exp(r*delta)-d)/(u-d)
    if(d >= math.exp(r*delta) or u <= math.exp(r*delta)):
        print("Arbitrage for M:", M)
    C=0
    for k in range(0,2**M):
        price = []
        price.append(S)
        cnt = 0
        for i in range(1,M+1):
            xx = 0
            if isKthBitSet(k,i):
                cnt+=1
                xx = price[-1]*u
            else:
                xx = price[-1]*d
            price.append(xx)
        S_max = 0.5*(np.max(price)+np.min(price))
        C_payoff=max(0,S_max-K)
        C+=math.pow(p,cnt) * math.pow((1-p),M-cnt) * C_payoff
    return C
    

def binomial_put(S, K, T, r, u, d, M):
    delta = T/M
    p = (math.exp(r*delta)-d)/(u-d)
    if(d >= math.exp(r*delta) or u <= math.exp(r*delta)):
        print("Arbitrage for M:", M)
    P=0
    for k in range(0,2**M):
        price = []
        price.append(S)
        cnt = 0
        for i in range(1,M+1):
            xx = 0
            if isKthBitSet(k,i):
                cnt+=1
                xx = price[-1]*u
            else:
                xx = price[-1]*d
            price.append(xx)
        S_max = 0.5*(np.max(price)+np.min(price))
        P_payoff=max(0,K-S_max)
        P+=math.pow(p,cnt) * math.pow((1-p),M-cnt) * P_payoff
    return P

delta=T/M
u=math.exp(sigma*math.sqrt(delta))
d=math.exp(-sigma*math.sqrt(delta))
call= binomial_call(S_0, k, T, r, u, d, M)
put= binomial_put(S_0, k, T, r, u, d, M)
print('Set-1')
print('Call option price :', call)
print('Put option price :',put)
u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
call= binomial_call(S_0, k, T, r, u, d, M)
put= binomial_put(S_0, k, T, r, u, d, M)
print('Set-2')
print('Call option price :', call)
print('Put option price :',put)

call_prices=[]
put_prices=[]
x = []
for i in range(0,1000,100):
    S_0=i
    delta=T/M
    u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
    d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
    call= binomial_call(S_0, k, T, r, u, d, M)
    put= binomial_put(S_0, k, T, r, u, d, M)
    call_prices.append(call)
    put_prices.append(put)
    x = np.linspace(0,1000,10)  
plt.plot(x, call_prices, label = "Call Option price")
plt.plot(x, put_prices, label = "Put Option price")
plt.title('Option Prices v/s $S{_0}$ Set-2')
plt.xlabel('$S{_0}$')
plt.ylabel('Option Prices')
plt.legend()
plt.show()
plt.clf() 
call_prices=[]
put_prices=[]
x = []
for i in range(0,1000,100):
    S_0=i
    delta=T/M
    u=math.exp(sigma*math.sqrt(delta))
    d=math.exp(-sigma*math.sqrt(delta))
    call= binomial_call(S_0, k, T, r, u, d, M)
    put= binomial_put(S_0, k, T, r, u, d, M)
    call_prices.append(call)
    put_prices.append(put)
    x = np.linspace(0,1000,10)  
plt.plot(x, call_prices, label = "Call Option price")
plt.plot(x, put_prices, label = "Put Option price")
plt.title('Option Prices v/s $S{_0}$ Set-1')
plt.xlabel('$S{_0}$')
plt.ylabel('Option Prices')
plt.legend()
plt.show()
plt.clf() 

S_0=100
call_prices=[]
put_prices=[]
x = []
for i in range(0,500,50):
    k=i
    delta=T/M
    u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
    d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
    call= binomial_call(S_0, k, T, r, u, d, M)
    put= binomial_put(S_0, k, T, r, u, d, M)
    call_prices.append(call)
    put_prices.append(put)
    x = np.linspace(0,1000,10)  
plt.plot(x, call_prices, label = "Call Option price")
plt.plot(x, put_prices, label = "Put Option price")
plt.title('Option Prices v/s k Set-2')
plt.xlabel('k')
plt.ylabel('Option Prices')
plt.legend()
plt.show()
plt.clf() 
call_prices=[]
put_prices=[]
x = []
for i in range(0,500,50):
    k=i
    delta=T/M
    u=math.exp(sigma*math.sqrt(delta))
    d=math.exp(-sigma*math.sqrt(delta))
    call= binomial_call(S_0, k, T, r, u, d, M)
    put= binomial_put(S_0, k, T, r, u, d, M)
    call_prices.append(call)
    put_prices.append(put)
    x = np.linspace(0,1000,10)  
plt.plot(x, call_prices, label = "Call Option price")
plt.plot(x, put_prices, label = "Put Option price")
plt.title('Option Prices v/s k Set-1')
plt.xlabel('k')
plt.ylabel('Option Prices')
plt.legend()
plt.show()
plt.clf() 

k=100
call_prices=[]
put_prices=[]
x = []
for i in range(1,51,5):
    r=i/100
    delta=T/M
    u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
    d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
    call= binomial_call(S_0, k, T, r, u, d, M)
    put= binomial_put(S_0, k, T, r, u, d, M)
    call_prices.append(call)
    put_prices.append(put)
    x = np.linspace(1,51,10)  
plt.plot(x, call_prices, label = "Call Option price")
plt.plot(x, put_prices, label = "Put Option price")
plt.title('Option Prices v/s r in % Set-2')
plt.xlabel('r')
plt.ylabel('Option Prices')
plt.legend()
plt.show()
plt.clf() 
call_prices=[]
put_prices=[]
x = []
for i in range(1,51,5):
    r=i/100
    delta=T/M
    u=math.exp(sigma*math.sqrt(delta))
    d=math.exp(-sigma*math.sqrt(delta))
    call= binomial_call(S_0, k, T, r, u, d, M)
    put= binomial_put(S_0, k, T, r, u, d, M)
    call_prices.append(call)
    put_prices.append(put)
    x = np.linspace(1,51,10)  
plt.plot(x, call_prices, label = "Call Option price")
plt.plot(x, put_prices, label = "Put Option price")
plt.title('Option Prices v/s r in % Set-1')
plt.xlabel('r')
plt.ylabel('Option Prices')
plt.legend()
plt.show()
plt.clf() 

r=0.08
call_prices=[]
put_prices=[]
x = []
for i in range(1,101,10):
    sigma=i/100
    delta=T/M
    u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
    d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
    call= binomial_call(S_0, k, T, r, u, d, M)
    put= binomial_put(S_0, k, T, r, u, d, M)
    call_prices.append(call)
    put_prices.append(put)
    x = np.linspace(0,100,10)  
plt.plot(x, call_prices, label = "Call Option price")
plt.plot(x, put_prices, label = "Put Option price")
plt.title('Option Prices v/s sigma in % Set-2')
plt.xlabel('sigma')
plt.ylabel('Option Prices')
plt.legend()
plt.show()
plt.clf() 
call_prices=[]
put_prices=[]
x = []
for i in range(21,121,10):
    sigma=i/100
    delta=T/M
    u=math.exp(sigma*math.sqrt(delta))
    d=math.exp(-sigma*math.sqrt(delta))
    call= binomial_call(S_0, k, T, r, u, d, M)
    put= binomial_put(S_0, k, T, r, u, d, M)
    call_prices.append(call)
    put_prices.append(put)
    x = np.linspace(0,100,10)  
plt.plot(x, call_prices, label = "Call Option price")
plt.plot(x, put_prices, label = "Put Option price")
plt.title('Option Prices v/s sigma in % Set-1')
plt.xlabel('sigma')
plt.ylabel('Option Prices')
plt.legend()
plt.show()
plt.clf()

sigma=.2

for i in [95,100,105]:
    k=i
    call_prices=[]
    put_prices=[]
    x = []
    for j in range(1,21,2):
        M=j
        delta=T/M
        u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
        d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
        call= binomial_call(S_0, k, T, r, u, d, M)
        put= binomial_put(S_0, k, T, r, u, d, M)
        call_prices.append(call)
        put_prices.append(put)
        x = np.linspace(0,20,10)
    plt.plot(x, call_prices, label = "Call Option price")
    plt.plot(x, put_prices, label = "Put Option price")
    plt.title('Option Prices v/s M for for k = '+str(i)+' Set-2' )
    plt.xlabel('M')
    plt.ylabel('Option Prices')
    plt.legend()
    plt.show()
    plt.clf()

for i in [95,100,105]:
    k=i
    call_prices=[]
    put_prices=[]
    x = []
    for j in range(1,21,2):
        M=j
        delta=T/M
        u=math.exp(sigma*math.sqrt(delta))
        d=math.exp(-sigma*math.sqrt(delta))
        call= binomial_call(S_0, k, T, r, u, d, M)
        put= binomial_put(S_0, k, T, r, u, d, M)
        call_prices.append(call)
        put_prices.append(put)
        x = np.linspace(0,20,10)
    plt.plot(x, call_prices, label = "Call Option price")
    plt.plot(x, put_prices, label = "Put Option price")
    plt.title('Option Prices v/s M for for k = '+str(i)+' Set-1' )
    plt.xlabel('M')
    plt.ylabel('Option Prices')
    plt.legend()
    plt.show()
    plt.clf()

k=100
M=10

x=np.linspace(0,1000,10)
y=np.linspace(0,500,10)
call_prices={}
put_prices={}
for i in range(0,10):
    for j in range(0,10):
        delta=T/M
        u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
        d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
        call= binomial_call(x[i], y[j], T, r, u, d, M)
        put= binomial_put(x[i], y[j], T, r, u, d, M)
        call_prices[i,j]=call
        put_prices[i,j]=put
ax = plt.axes(projection ='3d') 
ax.set_zlabel('Option Prices')
    
# plotting 
for i in range(0,10):
    for j in range(0,10):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying $S{_0}$ and k Set-2' )
ax.set_xlabel('$S{_0}$') 
ax.set_ylabel('k')

plt.show()  

x=np.linspace(0,1000,10)
y=np.linspace(0,500,10)
call_prices={}
put_prices={}
for i in range(0,10):
    for j in range(0,10):
        delta=T/M
        u=math.exp(sigma*math.sqrt(delta))
        d=math.exp(-sigma*math.sqrt(delta))
        call= binomial_call(x[i], y[j], T, r, u, d, M)
        put= binomial_put(x[i], y[j], T, r, u, d, M)
        call_prices[i,j]=call
        put_prices[i,j]=put
ax = plt.axes(projection ='3d') 
ax.set_zlabel('Option Prices')
    
# # plotting 
for i in range(0,10):
    for j in range(0,10):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying $S{_0}$ and k Set-1' ) 
ax.set_xlabel('$S{_0}$') 
ax.set_ylabel('k')
plt.show()  

k=100
S_0=100
x=np.linspace(0,1000,10)
y=np.linspace(1,51,10)
call_prices={}
put_prices={}
for i in range(0,10):
    for j in range(0,10):
        r=y[j]/100
        delta=T/M
        u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
        d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
        call= binomial_call(x[i], k, T, r, u, d, M)
        put= binomial_put(x[i], k, T, r, u, d, M)
        call_prices[i,j]=(call)
        put_prices[i,j]=(put)
ax = plt.axes(projection ='3d') 
ax.set_zlabel('Option Prices')
    
# # plotting 
for i in range(0,10):
    for j in range(0,10):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying $S{_0}$ and r in % Set-2' )
ax.set_xlabel('$S{_0}$') 
ax.set_ylabel('r')
plt.show()  

x=np.linspace(0,1000,10)
y=np.linspace(1,51,10)
call_prices={}
put_prices={}
for i in range(0,10):
    for j in range(0,10):
        delta=T/M
        u=math.exp(sigma*math.sqrt(delta))
        d=math.exp(-sigma*math.sqrt(delta))
        call= binomial_call(x[i], k, T, y[j]/100, u, d, M)
        put= binomial_put(x[i], k, T, y[j]/100, u, d, M)
        call_prices[i,j]=(call)
        put_prices[i,j]=(put)
ax = plt.axes(projection ='3d') 
ax.set_zlabel('Option Prices')
    
# # plotting 
for i in range(0,10):
    for j in range(0,10):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying $S{_0}$ and r in % Set-1' )
ax.set_xlabel('$S{_0}$') 
ax.set_ylabel('r')
plt.show()  

r=0.08
x=np.linspace(0,1000,10)
y=np.linspace(1,101,10)
call_prices={}
put_prices={}
for i in range(0,10):
    for j in range(0,10):
        sigma=y[j]/100
        delta=T/M
        u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
        d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
        call= binomial_call(x[i], k, T, r, u, d, M)
        put= binomial_put(x[i], k, T, r, u, d, M)
        call_prices[i,j]=(call)
        put_prices[i,j]=(put)
ax = plt.axes(projection ='3d') 
ax.set_zlabel('Option Prices')
    
# # plotting 
for i in range(0,10):
    for j in range(0,10):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying $S{_0}$ and sigma in % Set-2' ) 
ax.set_xlabel('$S{_0}$') 
ax.set_ylabel('sigma')
plt.show()  

x=np.linspace(0,1000,10)
y=np.linspace(1,101,10)
call_prices={}
put_prices={}
for i in range(0,10):
    for j in range(0,10):
        sigma=y[j]/100
        delta=T/M
        u=math.exp(sigma*math.sqrt(delta))
        d=math.exp(-sigma*math.sqrt(delta))
        call= binomial_call(x[i], k, T, r, u, d, M)
        put= binomial_put(x[i], k, T, r, u, d, M)
        call_prices[i,j]=(call)
        put_prices[i,j]=(put)
ax = plt.axes(projection ='3d') 
ax.set_zlabel('Option Prices')
    
# # plotting 
for i in range(0,10):
    for j in range(0,10):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying $S{_0}$ and sigma in % Set-1' ) 
ax.set_xlabel('$S{_0}$') 
ax.set_ylabel('sigma')
plt.show()

sigma=.2
x=np.linspace(0,1000,10)
y=np.linspace(1,11,10)
call_prices={}
put_prices={}
for i in range(0,10):
    for j in range(0,10):
        M=int(y[j])
        delta=T/M
        u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
        d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
        call= binomial_call(x[i], k, T, r, u, d, M)
        put= binomial_put(x[i], k, T, r, u, d, M)
        call_prices[i,j]=(call)
        put_prices[i,j]=(put)
ax = plt.axes(projection ='3d') 
ax.set_zlabel('Option Prices')
    
# # plotting 
for i in range(0,10):
    for j in range(0,10):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying $S{_0}$ and M Set-2' ) 
ax.set_xlabel('$S{_0}$') 
ax.set_ylabel('M')
plt.show()  

x=np.linspace(0,1000,10)
y=np.linspace(1,11,10)
call_prices={}
put_prices={}
for i in range(0,10):
    for j in range(0,10):
        M=int(y[j])
        delta=T/M
        u=math.exp(sigma*math.sqrt(delta))
        d=math.exp(-sigma*math.sqrt(delta))
        call= binomial_call(x[i], k, T, r, u, d, M)
        put= binomial_put(x[i], k, T, r, u, d, M)
        call_prices[i,j]=(call)
        put_prices[i,j]=(put)
ax = plt.axes(projection ='3d') 
ax.set_zlabel('Option Prices')
    
# # plotting 
for i in range(0,10):
    for j in range(0,10):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying $S{_0}$ and M Set-1' ) 
ax.set_xlabel('$S{_0}$') 
ax.set_ylabel('M')
plt.show()

M=10
S_0=100
x=np.linspace(1,501,10)
y=np.linspace(1,51,10)
call_prices={}
put_prices={}
for i in range(0,10):
    for j in range(0,10):
        k=x[i]
        r=y[j]/100
        delta=T/M
        u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
        d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
        call= binomial_call(S_0, k, T, r, u, d, M)
        put= binomial_put(S_0, k, T, r, u, d, M)
        call_prices[i,j]=(call)
        put_prices[i,j]=(put)
ax = plt.axes(projection ='3d') 
ax.set_zlabel('Option Prices')
    
# # plotting 
for i in range(0,10):
    for j in range(0,10):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying k and r in % Set-2' )
ax.set_xlabel('k') 
ax.set_ylabel('r') 
plt.show()  

x=np.linspace(1,501,10)
y=np.linspace(1,51,10)
call_prices={}
put_prices={}
for i in range(0,10):
    for j in range(0,10):
        k=x[i]
        r=y[j]/100
        delta=T/M
        u=math.exp(sigma*math.sqrt(delta))
        d=math.exp(-sigma*math.sqrt(delta))
        call= binomial_call(S_0, k, T, r, u, d, M)
        put= binomial_put(S_0, k, T, r, u, d, M)
        call_prices[i,j]=(call)
        put_prices[i,j]=(put)
ax = plt.axes(projection ='3d') 
ax.set_zlabel('Option Prices')
    
# # plotting 
for i in range(0,10):
    for j in range(0,10):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying k and r in % Set-1' )
ax.set_xlabel('k') 
ax.set_ylabel('r')  
plt.show()

r=0.08
x=np.linspace(1,501,10)
y=np.linspace(1,101,10)
call_prices={}
put_prices={}
for i in range(0,10):
    for j in range(0,10):
        k=x[i]
        sigma=y[j]/100
        delta=T/M
        u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
        d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
        call= binomial_call(S_0, k, T, r, u, d, M)
        put= binomial_put(S_0, k, T, r, u, d, M)
        call_prices[i,j]=(call)
        put_prices[i,j]=(put)
ax = plt.axes(projection ='3d') 
ax.set_zlabel('Option Prices')
    
# # plotting 
for i in range(0,10):
    for j in range(0,10):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying k and sigma in % Set-2' ) 
ax.set_xlabel('k') 
ax.set_ylabel('sigma') 
plt.show()  

x=np.linspace(1,501,10)
y=np.linspace(1,101,10)
call_prices={}
put_prices={}
for i in range(0,10):
    for j in range(0,10):
        k=x[i]
        sigma=y[j]/100
        delta=T/M
        u=math.exp(sigma*math.sqrt(delta))
        d=math.exp(-sigma*math.sqrt(delta))
        call= binomial_call(S_0, k, T, r, u, d, M)
        put= binomial_put(S_0, k, T, r, u, d, M)
        call_prices[i,j]=(call)
        put_prices[i,j]=(put)
ax = plt.axes(projection ='3d') 
ax.set_zlabel('Option Prices')
    
# # plotting 
for i in range(0,10):
    for j in range(0,10):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying k and sigma in % Set-1' ) 
ax.set_xlabel('k') 
ax.set_ylabel('sigma') 
plt.show()

sigma=0.2
x=np.linspace(1,501,10)
y=np.linspace(1,11,10)
call_prices={}
put_prices={}
for i in range(0,10):
    for j in range(0,10):
        k=x[i]
        M=int(y[j])
        delta=T/M
        u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
        d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
        call= binomial_call(S_0, k, T, r, u, d, M)
        put= binomial_put(S_0, k, T, r, u, d, M)
        call_prices[i,j]=(call)
        put_prices[i,j]=(put)
ax = plt.axes(projection ='3d') 
ax.set_zlabel('Option Prices')
    
# # plotting 
for i in range(0,10):
    for j in range(0,10):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying k and M Set-2' ) 
ax.set_xlabel('k') 
ax.set_ylabel('M') 
plt.show()  

x=np.linspace(1,501,10)
y=np.linspace(1,11,10)
call_prices={}
put_prices={}
for i in range(0,10):
    for j in range(0,10):
        k=x[i]
        M=int(y[j])
        delta=T/M
        u=math.exp(sigma*math.sqrt(delta))
        d=math.exp(-sigma*math.sqrt(delta))
        call= binomial_call(S_0, k, T, r, u, d, M)
        put= binomial_put(S_0, k, T, r, u, d, M)
        call_prices[i,j]=(call)
        put_prices[i,j]=(put)
ax = plt.axes(projection ='3d') 
ax.set_zlabel('Option Prices')
    
# # plotting 
for i in range(0,10):
    for j in range(0,10):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying k and M Set-1' ) 
ax.set_xlabel('k') 
ax.set_ylabel('M')  
plt.show()

M=10
k=100
x=np.linspace(1,101,10)
y=np.linspace(1,51,10)
call_prices={}
put_prices={}
for i in range(0,10):
    for j in range(0,10):
        sigma=x[i]/100
        r=y[j]/100
        delta=T/M
        u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
        d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
        call= binomial_call(S_0, k, T, r, u, d, M)
        put= binomial_put(S_0, k, T, r, u, d, M)
        call_prices[i,j]=(call)
        put_prices[i,j]=(put)
ax = plt.axes(projection ='3d') 
ax.set_zlabel('Option Prices')
    
# # plotting 
for i in range(0,10):
    for j in range(0,10):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying sigma in % and r in % Set-2' )
ax.set_xlabel('sigma') 
ax.set_ylabel('r') 
plt.show()  

x=np.linspace(51,101,10)
y=np.linspace(1,5,10)
call_prices={}
put_prices={}
for i in range(0,10):
    for j in range(0,10):
        sigma=x[i]/100
        r=y[j]/100
        delta=T/M
        u=math.exp(sigma*math.sqrt(delta))
        d=math.exp(-sigma*math.sqrt(delta))
        call= binomial_call(S_0, k, T, r, u, d, M)
        put= binomial_put(S_0, k, T, r, u, d, M)
        call_prices[i,j]=(call)
        put_prices[i,j]=(put)
ax = plt.axes(projection ='3d') 
ax.set_zlabel('Option Prices')
    
# # plotting 
for i in range(0,10):
    for j in range(0,10):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying sigma in % and r in % Set-1' ) 
ax.set_xlabel('sigma') 
ax.set_ylabel('r')
plt.show()

r=0.08
sigma=0.2
x=np.linspace(1,11,10)
y=np.linspace(1,51,10)
call_prices = {}
put_prices = {}
for i in range(0,10):
    for j in range(0,10):
        M=int(x[i])
        r=y[j]/100
        delta=T/M
        u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
        d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
        call= binomial_call(S_0, k, T, r, u, d, M)
        put= binomial_put(S_0, k, T, r, u, d, M)
        call_prices[(i,j)]=call
        put_prices[(i,j)]=put
ax = plt.axes(projection ='3d') 
ax.set_zlabel('Option Prices')
    
# plotting
for i in range(0,10):
    for j in range(0,10):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying M and r in % Set-2' ) 
ax.set_xlabel('M') 
ax.set_ylabel('r')
plt.show()  

x=np.linspace(2,12,10)
y=np.linspace(1,21,10)
call_prices={}
put_prices={}
for i in range(0,10):
    for j in range(0,10):
        M=int(x[i])
        r=y[j]/100
        delta=T/M
        u=math.exp(sigma*math.sqrt(delta))
        d=math.exp(-sigma*math.sqrt(delta))
        call= binomial_call(S_0, k, T, r, u, d, M)
        put= binomial_put(S_0, k, T, r, u, d, M)
        call_prices[i,j]=(call)
        put_prices[i,j]=(put)
ax = plt.axes(projection ='3d') 
ax.set_zlabel('Option Prices')
    
# plotting 
for i in range(0,10):
    for j in range(0,10):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying M and r in % Set-1' ) 
ax.set_xlabel('M') 
ax.set_ylabel('r')
plt.show()

r=0.08
x=np.linspace(1,11,10)
y=np.linspace(10,110,10)
call_prices={}
put_prices={}
for i in range(0,10):
    for j in range(0,10):
        M=int(x[i])
        sigma=y[j]/100
        delta=T/M
        u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
        d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
        call= binomial_call(S_0, k, T, r, u, d, M)
        put= binomial_put(S_0, k, T, r, u, d, M)
        call_prices[i,j]=(call)
        put_prices[i,j]=(put)
ax = plt.axes(projection ='3d') 
ax.set_zlabel('Option Prices')
    
# # plotting 
for i in range(0,10):
    for j in range(0,10):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying M and sigma in % Set-2' ) 
ax.set_xlabel('M') 
ax.set_ylabel('sigma')
plt.show()  

x=np.linspace(1,11,10)
y=np.linspace(10,110,10)
call_prices={}
put_prices={}
for i in range(0,10):
    for j in range(0,10):
        M=int(x[i])
        sigma=y[j]/100
        delta=T/M
        u=math.exp(sigma*math.sqrt(delta))
        d=math.exp(-sigma*math.sqrt(delta))
        call= binomial_call(S_0, k, T, r, u, d, M)
        put= binomial_put(S_0, k, T, r, u, d, M)
        call_prices[i,j]=(call)
        put_prices[i,j]=(put)
ax = plt.axes(projection ='3d') 
ax.set_zlabel('Option Prices')
    
# # plotting 
for i in range(0,10):
    for j in range(0,10):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying M and sigma in % Set-1' ) 
ax.set_xlabel('M') 
ax.set_ylabel('sigma')
plt.show()
