import math
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

def d_plus(S,K,T,r,sigma):
    return (math.log(S/K)+(r+0.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
def d_minus(S,K,T,r,sigma):
    return (math.log(S/K)+(r-0.5*sigma*sigma)*T)/(sigma*math.sqrt(T))

def bsm_call(S,K,T,t,r,sig):
    if(t==T):
        return np.maximum(0,S-K)
    return (S*norm.cdf(d_plus(S,K,T-t,r,sig)))-(K*math.exp(-r*(T-t))*norm.cdf(d_minus(S,K,T-t,r,sig)))
  
def bsm_put(S,K,T,t,r,sig):
    if(t==T):
        return np.maximum(0,K-S)
    return K*math.exp(-r*(T-t))-S+bsm_call(S,K,T,t,r,sig)


T=1
K=1
r=0.05
sigma=0.6
# print(bsm_call(0.5,K,T,0.2,r,sigma))
t=[0,0.2,0.4,0.6,0.8,1]
x=np.linspace(0.1,2.1,100)
# print(x)
for tt in t:
    bsm_c=[]
    for i in x:
        bsm_c.append(bsm_call(i,K,T,tt,r,sigma))
    plt.plot(x,bsm_c,label='t = '+ str(tt))
plt.title('C(t,x) Pricing according to bsm')
plt.xlabel('x')
plt.ylabel('Option Price')
plt.legend()
plt.show()

for tt in t:
    bsm_p=[]
    for i in x:
        bsm_p.append(bsm_put(i,K,T,tt,r,sigma))
    plt.plot(x,bsm_p,label='t = '+ str(tt))
plt.title('P(t,x) Pricing according to bsm')
plt.xlabel('x')
plt.ylabel('Option Price')
plt.legend()
plt.show()

bsm_c={}
bsm_p={}
xx=[]
TT=[]
bsm_cc=[]
bsm_pp=[]
for i in range(0,len(t)):
    for j in range(0,100):
        bsm_c[i,j]=bsm_call(x[j],K,T,t[i],r,sigma)
        bsm_p[i,j]=bsm_put(x[j],K,T,t[i],r,sigma)
        # if(bsm_p[i,j]<0):
        #     print(x[j],t[i])
        xx.append(x[j])
        TT.append(t[i])
        bsm_cc.append(bsm_c[i,j])
        bsm_pp.append(bsm_p[i,j])

ax = plt.axes(projection ='3d') 
# plotting 
# for i in range(0,len(t)):
#     for j in range(0,100):
#         ax.scatter(x[j], t[i], bsm_c[i,j], label='C(t,x)',color='cyan',s=0.5)

X = np.reshape(xx, (6, 100))
Y = np.reshape(TT, (6, 100))
Z = np.reshape(bsm_cc, (6, 100))
ax.plot_surface(X, Y, Z,cmap ='plasma', edgecolor ='pink')

ax.set_title('3D plot of C(t,x) varying t and x' )
ax.set_xlabel('x') 
ax.set_ylabel('t')
ax.set_zlabel('C(t,x)')
ax.view_init(40, 60)
# plt.legend()
plt.show()

ax = plt.axes(projection ='3d') 
# for i in range(0,len(t)):
#     for j in range(0,100):
#         ax.scatter(x[j], t[i], bsm_p[i,j], label='P(t,x)',color='orange',s=0.5) 
X = np.reshape(xx, (6, 100))
Y = np.reshape(TT, (6, 100))
Z = np.reshape(bsm_pp, (6, 100))
ax.view_init(40, 60)
ax.plot_surface(X, Y, Z, cmap ='viridis', edgecolor ='green')
ax.set_title('3D plot of P(t,x) varying t and x' )
ax.set_xlabel('x') 
ax.set_ylabel('t')
ax.set_zlabel('P(t,x)')
# plt.legend()
plt.show()

t=np.linspace(0,0.99,100)
bsm_c={}
bsm_p={}
xx=[]
TT=[]
bsm_cc=[]
bsm_pp=[]
for i in range(0,len(t)):
    for j in range(0,100):
        bsm_c[i,j]=bsm_call(x[j],K,T,t[i],r,sigma)
        bsm_p[i,j]=bsm_put(x[j],K,T,t[i],r,sigma)
        # if(bsm_c[i,j]<0):
        #     print(x[j],t[i])
        xx.append(x[j])
        TT.append(t[i])
        bsm_cc.append(bsm_c[i,j])
        bsm_pp.append(bsm_p[i,j])
# print(bsm_cc)
ax = plt.axes(projection ='3d') 
# plotting 
# for i in range(0,len(t)):
#     for j in range(0,100):
#         ax.scatter(x[j], t[i], bsm_c[i,j], label='C(t,x)',color='cyan',s=0.5)

X = np.reshape(xx, (100, 100))
Y = np.reshape(TT, (100, 100))
Z = np.reshape(bsm_cc, (100, 100))
ax.plot_surface(X, Y, Z,cmap ='plasma',edgecolor='pink')

ax.set_title('3D plot of C(t,x) varying t and x' )
ax.set_xlabel('x') 
ax.set_ylabel('t')
ax.set_zlabel('C(t,x)')
ax.view_init(40, 60)
# plt.legend()
plt.show()

ax = plt.axes(projection ='3d') 
# for i in range(0,len(t)):
#     for j in range(0,100):
#         ax.scatter(x[j], t[i], bsm_p[i,j], label='P(t,x)',color='orange',s=0.5) 
X = np.reshape(xx, (100, 100))
Y = np.reshape(TT, (100, 100))
Z = np.reshape(bsm_pp, (100, 100))
ax.view_init(40, 60)
ax.plot_surface(X, Y, Z, cmap ='viridis',edgecolor='green')
ax.set_title('3D plot of P(t,x) varying t and x' )
ax.set_xlabel('x') 
ax.set_ylabel('t')
ax.set_zlabel('P(t,x)')
# plt.legend()
plt.show()

