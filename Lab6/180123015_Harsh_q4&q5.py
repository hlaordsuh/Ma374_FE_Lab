import math
import numpy as np
import matplotlib.pyplot as plt  
import csv

def func(string,t):
    c = 1
    if t=="m":
        c = 61
    elif t=="w":
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
                arr=[]
                for val in row:
                    arr.append(val)
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
                
    for ar in S:
        for ele in ar:
            if ele == 0:
                ele = ar[0]
    return [S,arr]

tt=['m','w','d']
for h in tt:
    nn=50
    if(h=='d'):
        nn=985
    elif(h=='w'):
        nn=206
    ttt="monthly"
    if(h=='d'):
        ttt="daily"
    elif(h=='w'):
        ttt="weekly"
    names=["bsedata1","nsedata1","bsedata1_index_","nsedata1_index_"]
    cnt=0
    for name in names:
        w=10
        if cnt>1:
            w=1
        cnt+=1
        [S,arr]=func(name+h+".csv",h)                
        for q in range(0,w):
            stock=S[q][0:nn]
            n=len(stock)
            n1=len(S[q])-n
            
            u=[]
            #GENERATING Ui's
            #14 working days till 21st
            for i in range(1,n):
                if stock[i]==0:
                    stock[i]=stock[i-1]
                ui=np.log(stock[i]/stock[i-1])
                u.append(ui)

            E=np.mean(u)

            variance=0

            for i in u:
                t=(i-E)*(i-E)
                variance+=t

            variance/=(len(u)-1)
            mu=E+variance/2
            sigma=math.sqrt(variance)
            print("Estimated mu for "+arr[q+1]+" Stock calculated from "+ttt+" data : ",mu)
            print("Estimated sigma for "+arr[q+1]+" Stock calculated from "+ttt+" data : ",sigma)
            #Take 30th September to be the starting date, S_0 corresponds to the stock price on Sep 30, 2020
            S_0=stock[-1]
            #given values of lambda
            lamb_arr=[.005]

            for lamb in lamb_arr:
                #Generate N~ Poisson(lambda) for the counting process N(t)
                N=np.random.poisson(lamb,n1)
                #Generate Normal distribution 
                Z=np.random.normal(0,1,n1)
                #generate/simulate x and then exponentiate it to generate S
                X=[]
                X.append(math.log(S_0))
                for i in range(n1):
                    #if N=0 then M=0
                    M=0
                    #else M= sum(Y_j 1<=j<=N) where Y_j are distributed log-Normally
                    if(N[i]!=0):
                        LY=np.random.normal(mu,sigma,N[i])
                        M=np.sum(LY)
                    x=X[-1]+E+sigma*Z[i]+M
                    X.append(x)
                #Exponentiate X to get S
                SS=np.exp(X)

                # data to be plotted 
                # x = np.arange(1, 202)  
                y = np.array(SS) 
                x = np.arange(1, n1+2)
                # plotting 
                
                plt.title("Paths of "+arr[q+1]+" calculated from "+ttt+" data of Stock price " )  
                plt.xlabel("Time")  
                plt.ylabel("Stock price")  
                plt.plot(x, y, color ="red",label='simulated')
                
                plt.plot(x,S[q][nn-1:n1+n],color='blue',label='actual')
                plt.legend()
                plt.grid()  
                plt.show()
                
                        
