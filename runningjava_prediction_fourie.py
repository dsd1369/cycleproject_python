#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pymssql
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt

conn = pymssql.connect(server='sql16ssd-014.localnet.kr', user='i2on11_admin', password='root0826', database='i2on11_admin')
cursor = conn.cursor()

datas = {}
tank_seq = []
#tank_seq = [ '333','152','352']
for i in range(0,3):
    x_b = input('tank_seq : ')
    tank_seq.append(x_b)

for t_seq in tank_seq:
    conn_sql = 'SELECT signal_time, tank_remain_volume FROM gas_tank_volume_history where tank_seq = ' + t_seq + 'ORDER BY signal_time DESC'

    # 쿼리 실행
    cursor.execute(conn_sql)

    # 결과 가져오기
    rows = cursor.fetchall()
    datas[t_seq] = rows

# 연결 닫기
conn.close()


# In[2]:


d_h = {}

for it in tank_seq:
    d_h[it] = pd.DataFrame(datas[it], columns=['datetime', 'history'])
    d_h[it].set_index('datetime', inplace=True)


# In[3]:


# Differencing
diff_d_h = d_h.copy()
for it in tank_seq:
    diff_d_h[it] = diff_d_h[it]['history'].diff()
    diff_d_h[it] = diff_d_h[it].dropna()
#print('### Differenced Data ###') 
# for it in tank_seq:
#     #print(diff_d_h[it])


# In[4]:


#Differenced data plot
# plt.figure(figsize=(12,8)) 
# plt.subplot(211)
# plt.legend(['Raw Data (Nonstationary)'])
for it in tank_seq:
    diff_d_h[it][diff_d_h[it] >= 15] = 0
    #plt.plot(diff_d_h[it]) # first difference (t - (t-1)) plt.legend(['Differenced Data (Stationary)'])
#plt.show()


# In[5]:


diff_d_h_weekly={}
for it in tank_seq:
    diff_d_h_weekly[it] = -diff_d_h[it].resample('W').sum()
    diff_d_h_weekly[it] = diff_d_h_weekly[it].dropna()
    # print(diff_d_h_weekly)
    diff_d_h_weekly[it] = diff_d_h_weekly[it].drop(diff_d_h_weekly[it].index[-1])
    del diff_d_h_weekly[it][diff_d_h_weekly[it].index.min()]
    del diff_d_h_weekly[it][diff_d_h_weekly[it].index.max()]
    #diff_d_h_weekly[it].plot()


# In[6]:


import numpy as np
max_cycle_value = 150
diff_cycle = {}
for it in tank_seq:
    diff_cycle[it]=round(50*7/(diff_d_h_weekly[it]),2)
# print(diff_cycle)
    diff_cycle[it][np.isneginf(diff_cycle[it])] = max_cycle_value
    diff_cycle[it][diff_cycle[it] < 0] = max_cycle_value
    #diff_cycle[it].plot()


# In[7]:


import calendar

def get_days_in_month(year, month):
    return calendar.monthrange(year, month)[1]

def day_in_month_set(day_start_year,day_start_month, day_end_year, day_end_month):
    days_in_month = []
    if day_start_year > day_start_year or (day_start_year == day_end_year and day_start_month > day_end_month):
        print("옳지 않는 범위 설정")
        return days_in_month
    if day_start_year == day_end_year:
        days_in_month = [get_days_in_month(day_start_year, month) for month in range(day_start_month, day_end_month+1)]
    else:
        days_in_month = [get_days_in_month(day_start_year, month) for month in range(day_start_month, 13)]
        day_start_year = day_start_year+1
        while day_start_year < day_end_year:
            for month in range(1, 13):
                days_in_month.append(get_days_in_month(day_start_year, month))
                day_start_year = day_start_year+1
        for month in range(1, day_end_month+1):
                days_in_month.append(get_days_in_month(day_start_year, month))
    return days_in_month

def date_to_fraction(date, E_month):
    result = []
    days_in_month =  E_month
    #print(days_in_month)
    for d in date:
        year = d.year
        month = d.month
        day = d.day
        
        days_to_month = days_in_month[month - date.min().month]
        
        result.append(month + ((day-1)/days_to_month))
    return result


# In[8]:


import calendar

def get_days_in_month(year, month):
    return calendar.monthrange(year, month)[1]

def day_in_month_set(day_start_year,day_start_month, day_end_year, day_end_month):
    days_in_month = []
    if day_start_year > day_start_year or (day_start_year == day_end_year and day_start_month > day_end_month):
        print("옳지 않는 범위 설정")
        return days_in_month
    if day_start_year == day_end_year:
        days_in_month = [get_days_in_month(day_start_year, month) for month in range(day_start_month, day_end_month+1)]
    else:
        days_in_month = [get_days_in_month(day_start_year, month) for month in range(day_start_month, 13)]
        day_start_year = day_start_year+1
        while day_start_year < day_end_year:
            for month in range(1, 13):
                days_in_month.append(get_days_in_month(day_start_year, month))
                day_start_year = day_start_year+1
        for month in range(1, day_end_month+1):
                days_in_month.append(get_days_in_month(day_start_year, month))
    return days_in_month

def date_to_fraction(date, E_month):
    result = []
    days_in_month =  E_month
    #print(days_in_month)
    for d in date:
        year = d.year
        month = d.month
        day = d.day
        
        days_to_month = days_in_month[month - date.min().month]
        
        result.append(month + ((day-1)/days_to_month))
    return result


# In[10]:


import pandas as pd
import numpy as np
tr_F = {}
H_H = {}
W_W = {}
E_month = {}
for it in tank_seq:
    timelen = diff_cycle[it].index
    E_month[it] =  day_in_month_set(timelen.min().year, timelen.min().month, timelen.max().year, timelen.max().month)
    H = nontime_date = date_to_fraction(timelen, E_month[it])
    W = diff_cycle[it].values
    date_to_fraction(timelen ,E_month[it])
    nontime_date = pd.DataFrame(nontime_date)
    F = np.concatenate([[H], [W]])
    
    tr_F[it] = F.T
    H_H[it] = H
    W_W[it] = W


# In[11]:


import numpy as np
sdata  = {}
for it in tank_seq:
# Convert H and W to numpy arrays
    H = np.array(H_H[it])
    W = np.array(W_W[it])

# Sort H and W by ascending order of H
    sorted_indices = np.argsort(H)
    sorted_H = H[sorted_indices]
    sorted_W = W[sorted_indices]

# Transpose the arrays
    sdata[it] = F = np.vstack((sorted_H, sorted_W)).T
    #print(F)


# In[12]:


import matplotlib.pyplot as plt
from numpy import arange, ones, pi
from scipy import cos, sin
from scipy.fftpack import fft, fftfreq, ifft
# 주파수 구하기
yf_approx = {}
yf = {}
xf = {}
for it in tank_seq:
    N = sdata[it].shape[0]
    dt = sdata[it][1, 0] - sdata[it][0, 0]
    freq = np.fft.fftfreq(N, d=dt)

# 푸리에 변환하기
    yf[it] = fft(sdata[it][:, 1])
    xf[it] = freq

# 푸리에 근사치 구하기
    num_coeff = 10 # 사용할 계수 수
    yf_approx[it] = np.zeros_like


# In[13]:


y_approx = {}
for it in tank_seq:
    # 계수 추출
    yf_trunc = np.zeros_like(yf[it])
    yf_trunc[:num_coeff] = yf[it][:num_coeff]

    # 역변환
    y_approx[it] = np.real(ifft(yf_trunc))
#     print(sdata[it][:,0])
#     # 플롯
#     plt.plot(sdata[it][:, 0], sdata[it][:, 1], label='Original Data')
#     plt.plot(sdata[it][:, 0], y_approx[it], label='Fourier Approximation')
#     plt.legend()
#     plt.show()


# In[14]:


def fsl(x,a,b,c,d,e,f):
    return a*x**5+b*x**4+c*x**3+d*x**2+e*x+f
from scipy.optimize import curve_fit
import numpy as np
for it in tank_seq:
    H = y_approx[it]
    W =  sdata[it][:, 1]
    popt, pcov = curve_fit(fsl, H, W)

    modelf = lambda x: fsl(x, *popt)

#     plt.plot(sdata[it][:, 0], sdata[it][:, 1], label='Original Data')
#     #plt.plot(sdata[it][:, 0]+12, sdata[it][:, 1], label='Original Data')
#     #plt.plot(sdata[it][:, 0]-12, sdata[it][:, 1], label='Original Data')
    
#     plt.plot(sdata[it][:, 0], fsl(H, *popt), label='Fourier Approximation')
#     plt.legend()
#     plt.show()


# In[15]:


import calendar
def fslvalue_per_day(x0, x1, x2, itnum):
    day_in_month = get_days_in_month(x0, x1)
    x = x1 + x2/(day_in_month+1)
    it = tank_seq[itnum]
    #print(x, it)
    yfd = yf[it]
    yfd_trunc = np.zeros_like(yfd)
    yfd_trunc[:num_coeff] = yfd[:num_coeff]
    y_interp = np.interp(x, sdata[it][:, 0], np.real(ifft(yfd_trunc)))
    return y_interp

def remainby_cycle(x,remain):
    warning = [40, 30, 20]
    year, month, day, tank_seq_renum = x
    it = tank_seq[tank_seq_renum]
    while remain >=10:
        fsl_cycle_value = fsl(fslvalue_per_day(*x), *popt)
        if fsl_cycle_value < min(sdata[it][:,1]):
            fsl_cycle_value = min(sdata[it][:,1])
        used_gas = 50*7/fsl_cycle_value
        if remain>warning[0] and remain - used_gas <= warning[0]:
            print(warning[0],"Warning", year,".", month,".",day, " : ",round(remain - used_gas,3))
        elif remain>warning[1] and remain - used_gas <= warning[1]:
            print(warning[1],"Warning", year,".", month,".",day, " : ",round(remain - used_gas,3))
        elif remain>warning[2] and remain - used_gas <= warning[2]:
            print(warning[2],"Warning", year,".", month,".",day, " : ",round(remain - used_gas,3))
        remain = remain - used_gas
        max_days = calendar.monthrange(year, month)[1]
        day += 1
        if day > max_days:
            month += 1
            day = 1
            if month > 12:
                year += 1
                month = 1
        if remain <=10:
            break
        
    print(year,".", month,".",day, " : ",round(remain,3))
    x = [year,month,day, round(remain,3)]
    return x 


# In[16]:


# x = [2022, 8, 5, 0]
# remain = 80
x = []
q_x = ['year', 'month', 'day', 'list_num']
for i in range(0,4):
    x_a = input(q_x[i])
    x.append(int(x_a))
remain = input('잔량')
remain = int(remain)
remainby_cycle(x,remain)


# In[ ]:




