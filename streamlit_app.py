import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import matplotlib.dates as mdates
from datetime import datetime
from scipy.optimize import curve_fit
import os
import streamlit as st
import pandas as pd

#st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('台灣Covid-19疫情推估模式')
st.error('請注意：本模式僅作為學術討論用途，請勿轉載本站任何資訊或圖表，確切疫情資訊請以CDC為準！')
st.info('資料來源：國網中心 (https://covid-19.nchc.org.tw/)')
st.info('推估模式：Generalised logistic function (https://en.wikipedia.org/wiki/Generalised_logistic_function) (0504更新)')

date = []
diagnosed = []
total = []

res = requests.get('https://covid-19.nchc.org.tw/api/covid19?CK=covid-19@nchc.org.tw&querydata=4048')
data = res.json()

for i in data[::-1]:
	date.append(datetime.strptime(i['a01'], "%Y-%m-%d").date())
	diagnosed.append(float(i['a06']))
	if len(total) > 0:
		total.append(total[-1] + float(i['a06']))
	else:
		total.append(float(i['a06']))
	
x = np.arange(0, len(date))
today = len(date) - 1
start_day = 90
days = 243

def logistic_curve(x, a, b, c, d, i):
	return (a / np.power(1 + i * np.exp(-c * (x - d)), i)) + b
	
p0 = [total[-1], 1, 0.1, len(date) // 2, 2.5]
logistic_params, covariance = curve_fit(logistic_curve, x, total)
#print('logistic params=', logistic_params)

def plot1(x, logistic_params, covariance):
	plt.bar(x, diagnosed)
	plt.xlim([start_day, days])
	dtFmt = mdates.DateFormatter('%m-%d')
	plt.gca().xaxis.set_major_formatter(dtFmt) 
	plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
	plt.xticks(rotation=90, fontweight='light', fontsize='x-small',)
	plt.xlabel('Date')
	plt.ylabel('Number of Case')
	plt.title('COVID-19 Confirmed Case in Taiwan (2022)')
	x = np.arange(start_day, days)
	y = logistic_curve(x, *logistic_params) - logistic_curve(x-1, *logistic_params)
	plt.plot(x, y, color = 'r', label='Model')
	plt.legend(loc='best', fontsize=20)
	#plt.show()
	st.pyplot()

def plot2(x, logistic_params, covariance):
	plt.bar(x, total)
	plt.xlim([start_day, days])
	dtFmt = mdates.DateFormatter('%m-%d')
	plt.gca().xaxis.set_major_formatter(dtFmt) 
	plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
	plt.xticks(rotation=90, fontweight='light', fontsize='x-small',)
	plt.xlabel('Date')
	plt.ylabel('Number of Case')
	plt.title('COVID-19 Total Confirmed Case in Taiwan (2022)')
	x = np.arange(start_day, days)
	y = logistic_curve(x, *logistic_params)
	plt.plot(x, y, color = 'r', label='Model')
	plt.legend(loc='best', fontsize=20)
	#plt.show()
	st.pyplot()
	
def forcast(t):
	return logistic_curve(today + t, *logistic_params) - logistic_curve(today - 1 + t, *logistic_params)

st.warning('初始條件：' + str(date[0]) + '～' + str(date[-1]))

plot1(x, logistic_params, covariance)
plot2(x, logistic_params, covariance)

table = np.ones([22,3], dtype=np.uint32)

for i in range(1, 15):
	table[i-1, 0] = i
	table[i-1, 1] = forcast(i)
	table[i-1, 2] = logistic_curve(today + i, *logistic_params)

for i in range(1, 9):
	table[i+13, 0] = i * 15
	table[i+13, 1] = forcast(i * 15)
	table[i+13, 2] = logistic_curve(today + (i * 15), *logistic_params)

df = pd.DataFrame(
    table,
    columns=['預測日數', '推估確診', '累計確診'])
st.table(df)

hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
st.markdown(hide_table_row_index, unsafe_allow_html=True)