import matplotlib.pyplot as plt
import math
import numpy as np
from mpl_toolkits import mplot3d 

T=1
sigma=0.2
r=0.08
k=100
S_0=100
M=100

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
    return C[(0,0)]

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
    return P[(0,0)]

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
for i in range(0,1000,10):
    S_0=i
    delta=T/M
    u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
    d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
    call= binomial_call(S_0, k, T, r, u, d, M)
    put= binomial_put(S_0, k, T, r, u, d, M)
    call_prices.append(call)
    put_prices.append(put)
    x = np.linspace(0,1000,100)  
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
for i in range(0,1000,10):
    S_0=i
    delta=T/M
    u=math.exp(sigma*math.sqrt(delta))
    d=math.exp(-sigma*math.sqrt(delta))
    call= binomial_call(S_0, k, T, r, u, d, M)
    put= binomial_put(S_0, k, T, r, u, d, M)
    call_prices.append(call)
    put_prices.append(put)
    x = np.linspace(0,1000,100)  
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
for i in range(0,500,5):
    k=i
    delta=T/M
    u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
    d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
    call= binomial_call(S_0, k, T, r, u, d, M)
    put= binomial_put(S_0, k, T, r, u, d, M)
    call_prices.append(call)
    put_prices.append(put)
    x = np.linspace(0,1000,100)  
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
for i in range(0,500,5):
    k=i
    delta=T/M
    u=math.exp(sigma*math.sqrt(delta))
    d=math.exp(-sigma*math.sqrt(delta))
    call= binomial_call(S_0, k, T, r, u, d, M)
    put= binomial_put(S_0, k, T, r, u, d, M)
    call_prices.append(call)
    put_prices.append(put)
    x = np.linspace(0,1000,100)  
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
for i in range(1,51):
    r=i/100
    delta=T/M
    u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
    d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
    call= binomial_call(S_0, k, T, r, u, d, M)
    put= binomial_put(S_0, k, T, r, u, d, M)
    call_prices.append(call)
    put_prices.append(put)
    x = np.linspace(1,51,50)  
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
for i in range(1,51):
    r=i/100
    delta=T/M
    u=math.exp(sigma*math.sqrt(delta))
    d=math.exp(-sigma*math.sqrt(delta))
    call= binomial_call(S_0, k, T, r, u, d, M)
    put= binomial_put(S_0, k, T, r, u, d, M)
    call_prices.append(call)
    put_prices.append(put)
    x = np.linspace(1,51,50)  
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
for i in range(1,101):
    sigma=i/100
    delta=T/M
    u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
    d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
    call= binomial_call(S_0, k, T, r, u, d, M)
    put= binomial_put(S_0, k, T, r, u, d, M)
    call_prices.append(call)
    put_prices.append(put)
    x = np.linspace(0,100,100)  
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
for i in range(1,101):
    sigma=i/100
    delta=T/M
    u=math.exp(sigma*math.sqrt(delta))
    d=math.exp(-sigma*math.sqrt(delta))
    call= binomial_call(S_0, k, T, r, u, d, M)
    put= binomial_put(S_0, k, T, r, u, d, M)
    call_prices.append(call)
    put_prices.append(put)
    x = np.linspace(0,100,100)  
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
    for j in range(1,201):
        M=j
        delta=T/M
        u=math.exp(sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
        d=math.exp(-sigma*math.sqrt(delta)+delta*(r-0.5*sigma*sigma))
        call= binomial_call(S_0, k, T, r, u, d, M)
        put= binomial_put(S_0, k, T, r, u, d, M)
        call_prices.append(call)
        put_prices.append(put)
        x = np.linspace(0,200,200)
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
    for j in range(1,201):
        M=j
        delta=T/M
        u=math.exp(sigma*math.sqrt(delta))
        d=math.exp(-sigma*math.sqrt(delta))
        call= binomial_call(S_0, k, T, r, u, d, M)
        put= binomial_put(S_0, k, T, r, u, d, M)
        call_prices.append(call)
        put_prices.append(put)
        x = np.linspace(0,200,200)
    plt.plot(x, call_prices, label = "Call Option price")
    plt.plot(x, put_prices, label = "Put Option price")
    plt.title('Option Prices v/s M for for k = '+str(i)+' Set-1' )
    plt.xlabel('M')
    plt.ylabel('Option Prices')
    plt.legend()
    plt.show()
    plt.clf()

k=100
M=100

x=np.linspace(0,1000,20)
y=np.linspace(0,500,20)
call_prices={}
put_prices={}
for i in range(0,20):
    for j in range(0,20):
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
for i in range (0,20):
    for j in range (0,20):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying $S{_0}$ and k Set-2' )
ax.set_xlabel('$S{_0}$') 
ax.set_ylabel('k')

plt.show()  

x=np.linspace(0,1000,20)
y=np.linspace(0,500,20)
call_prices={}
put_prices={}
for i in range(0,20):
    for j in range(0,20):
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
for i in range (0,20):
    for j in range (0,20):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying $S{_0}$ and k Set-1' ) 
ax.set_xlabel('$S{_0}$') 
ax.set_ylabel('k')
plt.show()  

k=100
S_0=100
x=np.linspace(0,1000,20)
y=np.linspace(1,51,20)
call_prices={}
put_prices={}
for i in range(0,20):
    for j in range(0,20):
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
for i in range (0,20):
    for j in range (0,20):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying $S{_0}$ and r in % Set-2' )
ax.set_xlabel('$S{_0}$') 
ax.set_ylabel('r')
plt.show()  

x=np.linspace(0,1000,20)
y=np.linspace(1,51,20)
call_prices={}
put_prices={}
for i in range(0,20):
    for j in range(0,20):
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
for i in range (0,20):
    for j in range (0,20):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying $S{_0}$ and r in % Set-1' )
ax.set_xlabel('$S{_0}$') 
ax.set_ylabel('r')
plt.show()  

r=0.08
x=np.linspace(0,1000,20)
y=np.linspace(1,101,20)
call_prices={}
put_prices={}
for i in range(0,20):
    for j in range(0,20):
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
for i in range (0,20):
    for j in range (0,20):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying $S{_0}$ and sigma Set-2' ) 
ax.set_xlabel('$S{_0}$') 
ax.set_ylabel('sigma')
plt.show()  

x=np.linspace(0,1000,20)
y=np.linspace(1,101,20)
call_prices={}
put_prices={}
for i in range(0,20):
    for j in range(0,20):
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
for i in range (0,20):
    for j in range (0,20):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying $S{_0}$ and sigma Set-1' ) 
ax.set_xlabel('$S{_0}$') 
ax.set_ylabel('sigma')
plt.show()

sigma=.2
x=np.linspace(0,1000,20)
y=np.linspace(1,201,20)
call_prices={}
put_prices={}
for i in range(0,20):
    for j in range(0,20):
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
for i in range (0,20):
    for j in range (0,20):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying $S{_0}$ and M Set-2' ) 
ax.set_xlabel('$S{_0}$') 
ax.set_ylabel('M')
plt.show()  

x=np.linspace(0,1000,20)
y=np.linspace(1,201,20)
call_prices={}
put_prices={}
for i in range(0,20):
    for j in range(0,20):
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
for i in range (0,20):
    for j in range (0,20):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying $S{_0}$ and M Set-1' ) 
ax.set_xlabel('$S{_0}$') 
ax.set_ylabel('M')
plt.show()

M=100
S_0=100
x=np.linspace(1,501,20)
y=np.linspace(1,51,20)
call_prices={}
put_prices={}
for i in range(0,20):
    for j in range(0,20):
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
for i in range (0,20):
    for j in range (0,20):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying k and r in % Set-2' )
ax.set_xlabel('k') 
ax.set_ylabel('r') 
plt.show()  

x=np.linspace(1,501,20)
y=np.linspace(1,51,20)
call_prices={}
put_prices={}
for i in range(0,20):
    for j in range(0,20):
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
for i in range (0,20):
    for j in range (0,20):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying k and r in % Set-1' )
ax.set_xlabel('k') 
ax.set_ylabel('r')  
plt.show()

r=0.08
x=np.linspace(1,501,20)
y=np.linspace(1,101,20)
call_prices={}
put_prices={}
for i in range(0,20):
    for j in range(0,20):
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
for i in range (0,20):
    for j in range (0,20):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying k and sigma Set-2' ) 
ax.set_xlabel('k') 
ax.set_ylabel('sigma') 
plt.show()  

x=np.linspace(1,501,20)
y=np.linspace(1,101,20)
call_prices={}
put_prices={}
for i in range(0,20):
    for j in range(0,20):
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
for i in range (0,20):
    for j in range (0,20):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying k and sigma Set-1' ) 
ax.set_xlabel('k') 
ax.set_ylabel('sigma') 
plt.show()

sigma=0.2
x=np.linspace(1,501,20)
y=np.linspace(1,201,20)
call_prices={}
put_prices={}
for i in range(0,20):
    for j in range(0,20):
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
for i in range (0,20):
    for j in range (0,20):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying k and M Set-2' ) 
ax.set_xlabel('k') 
ax.set_ylabel('M') 
plt.show()  

x=np.linspace(1,501,20)
y=np.linspace(1,201,20)
call_prices={}
put_prices={}
for i in range(0,20):
    for j in range(0,20):
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
for i in range (0,20):
    for j in range (0,20):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying k and M Set-1' ) 
ax.set_xlabel('k') 
ax.set_ylabel('M')  
plt.show()

M=100
k=100
x=np.linspace(1,101,20)
y=np.linspace(1,51,20)
call_prices={}
put_prices={}
for i in range(0,20):
    for j in range(0,20):
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
for i in range (0,20):
    for j in range (0,20):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying sigma and r Set-2' )
ax.set_xlabel('sigma') 
ax.set_ylabel('r') 
plt.show()  

x=np.linspace(1,10,20)
y=np.linspace(1,10,20)
call_prices={}
put_prices={}
for i in range(0,20):
    for j in range(0,20):
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
for i in range (0,20):
    for j in range (0,20):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying sigma and r Set-1' ) 
ax.set_xlabel('sigma') 
ax.set_ylabel('r')
plt.show()

r=0.08
sigma=0.2
x=np.linspace(1,201,20)
y=np.linspace(1,51,20)
call_prices = {}
put_prices = {}
for i in range(0,20):
    for j in range(0,20):
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
for i in range (0,20):
    for j in range (0,20):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying M and r Set-2' ) 
ax.set_xlabel('M') 
ax.set_ylabel('r')
plt.show()  

x=np.linspace(2,202,20)
y=np.linspace(1,21,20)
call_prices={}
put_prices={}
for i in range(0,20):
    for j in range(0,20):
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
for i in range (0,20):
    for j in range (0,20):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying M and r Set-1' ) 
ax.set_xlabel('M') 
ax.set_ylabel('r')
plt.show()

r=0.08
x=np.linspace(1,201,20)
y=np.linspace(10,110,20)
call_prices={}
put_prices={}
for i in range(0,20):
    for j in range(0,20):
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
for i in range (0,20):
    for j in range (0,20):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying M and sigma Set-2' ) 
ax.set_xlabel('M') 
ax.set_ylabel('sigma')
plt.show()  

x=np.linspace(1,201,20)
y=np.linspace(10,110,20)
call_prices={}
put_prices={}
for i in range(0,20):
    for j in range(0,20):
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
for i in range (0,20):
    for j in range (0,20):
        ax.scatter(x[i], y[j], call_prices[i,j],label = "Call Option price",color='blue')
        ax.scatter(x[i], y[j], put_prices[i,j],label = "Put Option price",color='orange') 
ax.set_title('Varying M and sigma Set-1' ) 
ax.set_xlabel('M') 
ax.set_ylabel('sigma')
plt.show()
