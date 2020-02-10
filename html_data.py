# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import time
import requests
import sys
from bs4 import BeautifulSoup
from csv import writer 
import pandas as pd
#from csv import Dictwriter


with open('indep.csv','w') as file:
    csv_writer = writer(file)
    csv_writer.writerow(['T','TM','tm','H','P','VV','V','VM'])
    for year in range(2013,2019):
        for month in range(1,13):
            if(month<10):
                url ='https://en.tutiempo.net/climate/0{}-{}/ws-432950.html'.format(month,year)
            else:
                url = 'https://en.tutiempo.net/climate/{}-{}/ws-432950.html'.format(month,year)
                
            texts = requests.get(url)
            text_utf = texts.text.encode('utf=8')
            print(url)
            #with open('Data/Html_Data/{}\{}.html'.format(year,month),"wb") as output:
                     
                #output.write(text_utf)
                
            soup = BeautifulSoup(text_utf,"html.parser")
            days_of_month = soup.find_all(class_ = "medias mensuales numspan")
            
            #for day in range(1,32):  
            for tbody in days_of_month: 
                for day in range(1,32):
                    if tbody.find_all('tr')[day].find_all('td'):
                        T = tbody.find_all("tr")[day].find_all('td')[1].get_text()
                        TM = tbody.find_all("tr")[day].find_all('td')[2].get_text()
                        tm = tbody.find_all("tr")[day].find_all('td')[3].get_text()
                        H = tbody.find_all("tr")[day].find_all('td')[5].get_text()
                        P = tbody.find_all("tr")[day].find_all('td')[6].get_text()
                        VV= tbody.find_all("tr")[day].find_all('td')[7].get_text()
                        V = tbody.find_all("tr")[day].find_all('td')[8].get_text()
                        VM = tbody.find_all("tr")[day].find_all('td')[9].get_text()
                        
                        csv_writer.writerow([T,TM,tm,H,P,VV,V,VM]) 
                    
                    else:
                       break
                    #
                      
           
        
        sys.stdout.flush()
        
        
def first_df():       
    df_scrap = pd.read_csv('indep.csv')
    print(df_scrap)
        
if __name__ =="__main__":
     
    
    first_df()