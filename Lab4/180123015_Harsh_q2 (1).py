#Financial Engineering Lab 4 Question 2

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from mpl_toolkits import mplot3d

mu = np.array([0.1, 0.2, 0.15])
cov = np.array([[0.005, -0.010, 0.004], 
                [-0.010, 0.040, -0.002], 
                [0.004, -0.002, 0.023]])
dim = len(mu)
u = np.ones((1, dim))

def get_ret(w):
    return np.dot(w, mu)

def get_risk(w):
    return (np.matmul(np.matmul(w, cov), np.transpose(w)))**0.5

WT1 = []
WT2 = []
WT3 = []
def efficient_frontier(M):
    R = []
    for m in M:
        cons = (
            {'type': 'eq', 'fun': lambda w: np.sum(w)-1},
            {'type': 'eq', 'fun': lambda w: get_ret(w)-m}
        )
        bnds = ((0, 1), (0, 1), (0, 1))
        res = minimize(get_risk, np.array([0.2, 0.3, 0.5]), method='SLSQP', bounds=bnds, constraints=cons)
        R.append(res.fun)
        WT1.append(res.x[0])
        WT2.append(res.x[1])
        WT3.append(res.x[2])
    return R

def minimum_variance(M, i):
    R = []
    W = []
    for m in M:
        cons = (
            {'type': 'eq', 'fun': lambda w: w[i-1]},
            {'type': 'eq', 'fun': lambda w: np.sum(w)-1},
            {'type': 'eq', 'fun': lambda w: get_ret(w)-m}
        )
        bnds = ((0, 1), (0, 1), (0, 1))
        res = minimize(get_risk, np.array([0, 0, 1]), method='SLSQP', bounds=bnds, constraints=cons)
        R.append(res.fun)
        if i == 1:
            W.append([res.x[1], res.x[2]])
        if i == 2:
            W.append([res.x[0], res.x[2]])
        if i == 3:
            W.append([res.x[0], res.x[1]])
    return R, W

M = np.linspace(0.1, 0.2, 100)
R = efficient_frontier(M)

M1 = np.linspace(0.1, 0.2, 100)
R1, W1 = minimum_variance(M1, 3)
W1 = np.transpose(W1)

M2 = np.linspace(0.1, 0.15, 100)
R2, W2 = minimum_variance(M2, 2)
W2 = np.transpose(W2)

M3 = np.linspace(0.15, 0.2, 100)
R3, W3 = minimum_variance(M3, 1)
W3 = np.transpose(W3)

w1 = 0
Y = []
X = []
while w1 <= 1:
    w2 = 0
    while w2 <= 1-w1:
        w3 = 1 - w1 - w2
        w = np.array([w1, w2, w3])
        m = np.dot(w, mu)
        r = (np.matmul(np.matmul(w, cov), np.transpose(w)))**0.5
        Y.append(m)
        X.append(r)
        w2 += 0.01
    w1 += 0.01

plt.scatter(X, Y, s = 1, color='pink',linestyle='-.', label='Feasible Region')

plt.plot(R, M, color='red', linestyle='-.',label='Minimum Variance Curve')
plt.plot(R, M, color='red', label='Efficient Frontier')
plt.plot(R1, M1, color='blue', linestyle='dashed',label='Minimum Variance Curve With securities 2 and 3')
plt.plot(R2, M2, color='orange',linestyle='dashed', label='Minimum Variance Curve With securities 1 and 3')
plt.plot(R3, M3, color='green',linestyle='dashed', label='Minimum Variance Curve With securities 1 and 2')
plt.title('Feasible Region & Efficient Frontier (Without Short Sales)')
plt.xlabel('Volatility(${\sigma}$)')
plt.ylabel('Return value(${\mu}$)')
plt.legend()
plt.show()

ax = plt.axes(projection ='3d') 
ax.plot3D(WT1, WT2, WT3, 'green') 
ax.set_title('Weights Corresponding to Minimum Variance Curve')
ax.set_xlabel('Weight 1')
ax.set_ylabel('Weight 2')
ax.set_zlabel('Weight 3')
plt.show()

plt.plot(WT1,WT2)
plt.title('Weights Corresponding to Minimum Variance Curve')
plt.xlabel('Weight 1')
plt.ylabel('Weight 2')
plt.show()