# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 18:15:56 2020

@author: Nikitha
"""

import pandas as pd
import matplotlib.pyplot as plt

def avg_day():
    average =[]   
    for year in range(2013,2019):
           
        
        for rows in pd.read_csv(r'C:\Users\Nikitha\.spyder-py3\aqi{}.csv'.format(year),chunksize=24):
            data=[]
            var=0
            avg=0.0
            avg_d =[]
            print(rows)
            temp_i=0
            df = pd.DataFrame(data=rows)
            print(df)
            for index,row in df.iterrows():
                data.append(row['PM2.5'])
                #print(data[:30])
            for i in data:
                if type(i) is int or type(i) is float:
                    var = var + i
                elif type(i) is str:
                    if i!='NoData' and i!='PwrFail' and i!='---' and i!='InVld':
                        temp = float(i)
                        var = var + temp
                    #print(var)
            avg = var/24
                   # print(avg)
            #temp_i = temp_i+1
            average.append(avg)
            
    print(len(average))
    df_avg = pd.DataFrame(average,columns=['AQI'])
    print(df_avg)


if __name__ == "__main__":
    avg_day()