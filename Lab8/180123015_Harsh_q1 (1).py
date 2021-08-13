import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from scipy.stats import norm

TRADING_DAYS = 252

nse_name = ['HDFCBANK.NS', 'TATAMOTORS.NS', 'ADANIENT.NS', 'VOLTAS.NS', 'BAJAJFINSV.NS', 'HAVELLS.NS', 'RELIANCE.NS', 'BERGEPAINT.NS', 'ASIANPAINT.NS', 'MUTHOOTFIN.NS']
bse_name = ['ASTRAL.BO', 'JINDALSTEL.BO', 'JUBLINDS.BO', 'VOLTAS.BO', 'JUBLINDS.BO', 'HAVELLS.BO', 'RELIANCE.BO', 'BOMDYEING.BO', 'BERGEPAINT.BO', 'MANAPPURAM.BO']

def func_index(string,st):
    c = 1230
    S = np.zeros(shape=(st,c-1))
    with open(string) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        lis = list(csv_reader)
        line_count = 1
        for row in lis[1:]:
            if line_count == c:
                break
            
            cn = 0
            for val in row[1:]:
                if val=='':
                    continue
                S[cn][line_count-1] = float(val)
                cn=cn+1
            
            line_count += 1
    
    arr = np.zeros(shape=(st,21))
    for i in range(0,st):
        arr[i] = S[i][-21:]
    return arr

bse_stocks = func_index("bsedata1_daily.csv",10)
nse_stocks = func_index("nsedata1_daily.csv",10)
bse_index = func_index("bsedata1_index_daily.csv",1)
nse_index = func_index("nsedata1_index_daily.csv",1)

def annual_volatility(daily_prices):
    prices = daily_prices
    rets=[]
    for i in range(1,len(prices)):
        rets.append((prices[i]-prices[i-1])/(prices[i-1]))
    return np.std(rets)*math.sqrt(TRADING_DAYS)


print("****************** Question 1 *********************")


print("For BSE Stocks")
header = f' {"Stock":20s}  {"Volatility (1 month)":15s} '
print('\n' + header)
# print('\n' + header + '\n'+ '-'*len(header))
for i in range(0,len(bse_stocks)):
    prices = bse_stocks[i]
    vol = annual_volatility(prices)
    print(f' {bse_name[i]:20s}  {vol:16.9f}     ')
    
print("")
print("")
print("For NSE Stocks")
header = f' {"Stock":20s}  {"Volatility (1 month)":15s} '
print('\n' + header)
# print('\n' + header + '\n'+ '-'*len(header))
for i in range(0,len(nse_stocks)):
    prices = nse_stocks[i]
    vol = annual_volatility(prices)
    print(f' {nse_name[i]:20s}  {vol:16.9f}     ')
    
print("")
print("")
print("For SENSEX")
header = f' {"Index":20s}  {"Volatility (1 month)":15s} '
# print('\n' + header + '\n'+ '-'*len(header))
print('\n' + header)
for i in range(0,len(bse_index)):
    prices = bse_index[i]
    vol = annual_volatility(prices)
    print(f' {"SENSEX":20s}  {vol:16.9f}     ')
    
print("")
print("")
print("For NIFTY")
header = f' {"Index":20s}  {"Volatility (1 month)":15s} '
# print('\n' + header + '\n'+ '-'*len(header))
print('\n' + header)
for i in range(0,len(nse_index)):
    prices = nse_index[i]
    vol = annual_volatility(prices)
    print(f' {"NIFTY":20s}  {vol:16.9f}     ')
    
    



