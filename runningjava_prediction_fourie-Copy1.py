#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pymssql
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt

def server_connection(tank_seq):
    datas = {}
    conn = pymssql.connect(server='sql16ssd-014.localnet.kr', user='i2on11_admin', password='root0826', database='i2on11_admin')
    cursor = conn.cursor()
    for it in tank_seq:
        conn_sql = 'SELECT signal_time, tank_remain_volume FROM gas_tank_volume_history where tank_seq = ' + str(it) + 'ORDER BY signal_time DESC'
        # 쿼리 실행
        cursor.execute(conn_sql)

        # 결과 가져오기
        rows = cursor.fetchall()
        datas[it] = rows
    # 연결 닫기
    conn.close()
    return datas


# In[80]:


def datas_to_diff_cycle_data(datas, tank_seq):
    diff_cycle_it = {}
    max_cycle_value = 150
    for it in tank_seq:
        d_h = pd.DataFrame(datas[it], columns=['datetime', 'history'])
        d_h.set_index('datetime', inplace=True)
        diff_d_h = d_h.copy()
        diff_d_h = diff_d_h['history'].diff()
        diff_d_h = diff_d_h.dropna()
        diff_d_h[diff_d_h >= 15] = 0
        
        diff_d_h_weekly = -diff_d_h.resample('W').sum()
        diff_d_h_weekly = diff_d_h_weekly.dropna()
        diff_d_h_weekly = diff_d_h_weekly.drop(diff_d_h_weekly.index[-1])
        
        del diff_d_h_weekly[diff_d_h_weekly.index.min()]
        del diff_d_h_weekly[diff_d_h_weekly.index.max()]
        
        diff_cycle = round(50*7/(diff_d_h_weekly),2)
        diff_cycle[np.isneginf(diff_cycle)] = max_cycle_value
        diff_cycle[diff_cycle < 0] = max_cycle_value
        diff_cycle_it[it] = diff_cycle
    return diff_cycle_it


# In[89]:


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


# In[90]:


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


# In[99]:


import pandas as pd
import numpy as np
def make_non_timeseries(diff_cycle):
    sdata  = {}
    for it in tank_seq:
        timelen = diff_cycle[it].index
        E_month =  day_in_month_set(timelen.min().year, timelen.min().month, timelen.max().year, timelen.max().month)
        H = nontime_date = date_to_fraction(timelen, E_month)
        W = diff_cycle[it].values
        date_to_fraction(timelen ,E_month)
        nontime_date = pd.DataFrame(nontime_date)
        F = np.concatenate([[H], [W]])
    
        tr_F = F.T
        H_H = H
        W_W = W
        H = np.array(H_H)
        W = np.array(W_W)

        # Sort H and W by ascending order of H
        sorted_indices = np.argsort(H)
        sorted_H = H[sorted_indices]
        sorted_W = W[sorted_indices]

        # Transpose the arrays
        sdata[it] = F = np.vstack((sorted_H, sorted_W)).T
        #print(F)
    return sdata


# In[144]:


import matplotlib.pyplot as plt
from numpy import arange, ones, pi
from scipy import cos, sin
from scipy.fftpack import fft, fftfreq, ifft
from scipy.optimize import curve_fit
import numpy as np
import pickle


def fsl(x,a,b,c,d,e,f):
    return a*x**5+b*x**4+c*x**3+d*x**2+e*x+f

def fourie_fitting(sdata):
    # 주파수 구하기
    popt_set = {}
    yf = {}
    num_coeff = 10
    for it in tank_seq:
        N = sdata[it].shape[0]
        dt = sdata[it][1, 0] - sdata[it][0, 0]
        freq = np.fft.fftfreq(N, d=dt)

    # 푸리에 변환하기
        yf[it] = fft(sdata[it][:, 1])
        xf = freq

    # 푸리에 근사치 구하기
    # 사용할 계수 수
        yf_approx = np.zeros_like
        yf_trunc = np.zeros_like(yf[it])
        yf_trunc[:num_coeff] = yf[it][:num_coeff]

        # 역변환
        y_approxd = np.real(ifft(yf_trunc))
        
        H = y_approxd
        W =  sdata[it][:, 1]
        popt, pcov = curve_fit(fsl, H, W)

        modelf = lambda x: fsl(x, *popt)
        popt_set[it] = popt
    return popt_set, yf, num_coeff


# In[163]:


import calendar
from scipy.fftpack import fft, fftfreq, ifft
def fslvalue_per_day(x0, x1, x2, it, yf, sdata):
    day_in_month = get_days_in_month(x0, x1)
    x = x1 + x2/(day_in_month+1)
    yfd = yf[it]
    num_coeff = 10
    yfd_trunc = np.zeros_like(yfd)
    yfd_trunc[:num_coeff] = yfd[:num_coeff]
    y_interp = np.interp(x, sdata[it][:, 0], np.real(ifft(yfd_trunc)))
    return y_interp

def remainby_cycle(x,remain, popt_set, yf, sdata):
    warning = [40, 30, 20]
    year, month, day, it = x
    while remain >=10:
        fsl_cycle_value = fsl(fslvalue_per_day(year, month, day, it, yf, sdata), *popt_set[it])
        if fsl_cycle_value < min(sdata[it][:,1]):
            fsl_cycle_value = min(sdata[it][:,1])
        used_gas = 50*7/fsl_cycle_value
        if remain > warning[0] and remain - used_gas <= warning[0]:
            print(warning[0],"Warning", year,".", month,".",day, " : ",round(remain - used_gas,3))
        elif remain > warning[1] and remain - used_gas <= warning[1]:
            print(warning[1],"Warning", year,".", month,".",day, " : ",round(remain - used_gas,3))
        elif remain > warning[2] and remain - used_gas <= warning[2]:
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


# In[184]:


def runmodel(seq):
    tank_seq = ['333','152','352']
    datas = server_connection(tank_seq_list)
    datas_to_diff_cycle = datas_to_diff_cycle_data(datas, tank_seq)
    sdata = make_non_timeseries(datas_to_diff_cycle)
    fourie = fourie_fitting(sdata)
    popt_set = fourie[0]
    yf = fourie[1]
    x = [2022, 8, 5, seq]
    remain = 80

    return remainby_cycle(x,remain, popt_set, yf, sdata)


# In[185]:


print(runmodel('333'))


# In[ ]:




