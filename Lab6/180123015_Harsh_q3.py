import numpy as np
import math
import numpy as np
import csv
import matplotlib.pyplot as plt

nse_name = ['HDFCBANK.NS', 'TATAMOTORS.NS', 'ADANIENT.NS', 'VOLTAS.NS', 'BAJAJFINSV.NS', 'HAVELLS.NS', 'RELIANCE.NS', 'BERGEPAINT.NS', 'ASIANPAINT.NS', 'MUTHOOTFIN.NS']
bse_name = ['ASTRAL.BO', 'JINDALSTEL.BO', 'JUBLINDS.BO', 'VOLTAS.BO', 'JUBLINDS.BO', 'HAVELLS.BO', 'RELIANCE.BO', 'BOMDYEING.BO', 'BERGEPAINT.BO', 'MANAPPURAM.BO']

def func_index(string,t,cent):
    c = 1
    if t=="Months":
        c = 61
    elif t=="Weeks":
        c = 261
    else:
        c = 1229
    
    S = np.zeros(shape=(1,c-1))
    with open(string) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == c:
                break
            if line_count == 0:
                line_count += 1
            else:
                cn = -1
                for val in row:
                    if cn==-1:
                        cn+=1
                        continue
                    if val=='':
                        continue
                    S[cn][line_count-1] = float(val)
                    cn=cn+1
                line_count += 1
                
    for ar in S:
        for ele in ar:
            if ele == 0:
                ele = ar[0]
                
    for ar in S:
        u = []
        for i in range(1,len(ar)):
            u.append(math.log(ar[i]/ar[i-1]))
        n = len(u)
        Eu = 0 
        for i in range(0,len(u)):
            Eu += u[i]
        Eu = Eu/(n)
        sigma = 0 
        for i in range(0,len(u)):
            sigma += (u[i]-Eu)*(u[i]-Eu)
        sigma = sigma/(n-1)
        sigma = math.sqrt(sigma)
        mu = 0
        mu = Eu + (sigma*sigma)/2
        
        log_ret_norm = []
        for i in range(0,len(u)):
            log_ret_norm.append(((u[i] - mu)/sigma))
        _, bins, _ = plt.hist(log_ret_norm, density=True, bins=25)
        X = np.linspace(bins[0], bins[-1])
        plt.plot(X, np.exp(-X**2 / 2)/(np.sqrt(np.pi*2)))
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.title("Density Plot of "+cent+" index returns " + "("+t+")" )
        plt.show()
        
            

    

def func(string,arr,t,cent):
    c = 1
    if t=="Months":
        c = 61
    elif t=="Weeks":
        c = 261
    else:
        c = 1229
    
    S = np.zeros(shape=(10,c-1))
    with open(string) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == c:
                break
            if line_count == 0:
                line_count += 1
            else:
                
                cn = -1
                for val in row:
                    if cn==-1:
                        cn+=1
                        continue
                    if val=='' or val=='None':
                        continue
                    S[cn][line_count-1] = float(val)
                    cn=cn+1
                line_count += 1
                
    cn = 0
    for ar in S:
        u = []
        for i in range(1,len(ar)):
            if(ar[i]==0):
                ar[i]=ar[i-1]
            u.append(math.log(ar[i]/ar[i-1]))
        n = len(u)
        Eu = 0 
        for i in range(0,len(u)):
            Eu += u[i]
        Eu = Eu/(n)
        sigma = 0 
        for i in range(0,len(u)):
            sigma += (u[i]-Eu)*(u[i]-Eu)
        sigma = sigma/(n-1)
        sigma = math.sqrt(sigma)
        mu = 0
        mu = Eu + (sigma*sigma)/2
        
        log_ret_norm = []
        for i in range(0,len(u)):
            log_ret_norm.append(((u[i] - mu)/sigma))
        _, bins, _ = plt.hist(log_ret_norm, density=True, bins=25)
        X = np.linspace(bins[0], bins[-1])
        plt.plot(X, np.exp(-X**2 / 2)/(np.sqrt(np.pi*2)))
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.title("Density Plot of " + arr[cn] + "(" + cent + "-"+t + ")")
        plt.show()
        cn+=1
    
        

print("Monthly")
print("*****************************")
func_index('bsedata1_index_m.csv',"Months","bse")
print("*****************************")
func_index('nsedata1_index_m.csv',"Months","nse")
print("*****************************")
func('bsedata1m.csv',bse_name,"Months","bse")
print("*****************************")
func('nsedata1m.csv',nse_name,"Months","nse")

print("Weekly")
print("*****************************")
func_index('bsedata1_index_w.csv',"Weeks","bse")
print("*****************************")
func_index('nsedata1_index_w.csv',"Weeks","nse")
print("*****************************")
func('bsedata1w.csv',bse_name,"Weeks","bse")
print("*****************************")
func('nsedata1w.csv',nse_name,"Weeks","nse")

print("Daily")
print("*****************************")
func_index('bsedata1_index_d.csv',"Days","bse")
print("*****************************")
func_index('nsedata1_index_d.csv',"Days","nse")
print("*****************************")
func('bsedata1d.csv',bse_name,"Days","bse")
print("*****************************")
func('nsedata1d.csv',nse_name,"Days","nse")


