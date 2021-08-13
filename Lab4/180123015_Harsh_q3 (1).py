import math
import numpy as np
import csv
import matplotlib.pyplot as plt

S = np.zeros(shape=(10,60))

with open('Price_Data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 61:
            break
        if line_count == 0:
            line_count += 1
        else:
            cn = -1
            for val in row:
                if cn==-1:
                    cn+=1
                    continue
                S[cn][line_count-1] = float(val)
                cn=cn+1
                
            line_count += 1
            
m = np.array([[0,0,0,0,0,0,0,0,0,0]])
u = np.array([[1,1,1,1,1,1,1,1,1,1]])
C = np.zeros(shape=(10,10))

cn = 0
for arr in S:
    m[0][cn] = arr.mean()
    cn+=1
    
for i in range(0,10):
    for j in range(0,10):
        val = 0
        for k in range(0,60):
            val+= (S[i][k]-m[0][i])*(S[j][k]-m[0][j])
        C[i][j] = val/60
        
print("Mean return :",m)
print("Covariance matrix:",C)
        
C_inv = np.linalg.inv(C)
mu_arr = np.linspace(-100,700,401)
xx1 = []
SD1_arr = []

M11 = (m.dot(C_inv)).dot(m.T)
M12 = (u.dot(C_inv)).dot(m.T)
M21 = (m.dot(C_inv)).dot(u.T)
M22 = (u.dot(C_inv)).dot(u.T)

M = np.array([[M11[0][0],M12[0][0]],[M21[0][0],M22[0][0]]])
M_inv = np.linalg.inv(M)



for mu in mu_arr:
    lamda = 2*M_inv.dot((np.array([[mu,1]])).T)
    w = (lamda[0][0]* (m.dot(C_inv)) + (lamda[1][0]*(u.dot(C_inv))))/2
    var = w.dot(C.dot(w.T))
    xx1.append(mu)
    SD1_arr.append(math.sqrt(var[0][0]))
    
# Part A

plt.plot(SD1_arr, xx1,c='r',label = "Markowitz Efficient Frontier")
plt.xlabel("Standard Variation (${\sigma}$)")
plt.ylabel("Return Value (${\mu}$)")
plt.title("Markowitz efficient Frontier")
plt.grid(True)
plt.legend()
plt.show()

# Part B
rf = 0.05

num = (m-rf*u).dot(C_inv)
denom = (m-rf*u).dot(C_inv.dot(u.T))
w = num/denom
mu = w.dot(m.T)
var = w.dot(C.dot(w.T))

print("")
print("For a 5% risk free return, the return ",round((mu[0][0]),4))
print("For a 5% risk free return, the risk ",round(math.sqrt(var[0][0]),4))
print("w1 =",round(w[0][0],4),", w2 =",round(w[0][1],4),", w3 =",round(w[0][2],4),", w4 =",round(w[0][3],4),", w5 =",round(w[0][4],4))
print("w6 =",round(w[0][5],4),", w7 =",round(w[0][6],4),", w8 =",round(w[0][7],4),", w9 =",round(w[0][8],4),", w10 =",round(w[0][9],4))

# Part C

point1 = (0,rf)
point2 = (math.sqrt(var[0][0]),mu[0][0])

x_values = [point1[0],point2[0],45]

y_values = [point1[1], point2[1],rf+45*(point2[1]-rf)/point2[0]]


plt.plot(x_values, y_values,label="CML")
plt.plot(SD1_arr, xx1,c='r',label = "Markowitz Efficient Frontier")
plt.scatter([math.sqrt(var[0][0])],[mu[0][0]],c='g',label="Market Portfolio")
plt.xlabel("Standard Variation (${\sigma}$)")
plt.ylabel("Return Value (${\mu}$)")
plt.yticks(np.linspace(-100,700,11))
plt.xticks(np.linspace(0,40,11))
plt.title("Markowitz efficient Frontier and CML")
plt.grid(True)
plt.legend()
plt.show()

# Part D
stocks = ["Google","Airtel","Microsoft","Apple","Tesla"	,"SBI"	,"Amazon"	,"Infosys"	,"Facebook"	,"Reliance"]
for i in range(0,10):
    beta = np.linspace(-2,2,401)
    mu_v = rf + (m[0][i] - rf)*beta

    plt.plot(beta, mu_v,label = stocks[i])

    plt.xlabel("Beta Coefficient (${\ beta}$)")
    plt.ylabel("Return Value (${\mu}$)")
    plt.title("Security Market Line")
    plt.legend()
    plt.grid(True)
    
plt.show()


