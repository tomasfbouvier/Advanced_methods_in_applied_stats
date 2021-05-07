# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 15:26:21 2021

@author: To+
"""
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pdfrw 
from tika import parser 
import re

def rmv_numb(string):
    #uglish horrible method for removing undesirable numbers for pandas begginer
    string = ''.join(i for i in string if not i.isdigit()).rstrip()
    return string

def Convert(string): 
    li = list(string.split(",")) 
    return li 

def rmv_numb2(lista):
    #uglish horrible method for removing undesirable numbers for pandas begginer
    for j in range(len(lista)):
        lista[j] = ''.join(i for i in lista[j] if not i.isdigit()).rstrip()
    return lista

def nosemeocurrenombre(html):
    with open(html) as fp:
        soup = BeautifulSoup(fp, "html.parser")
    table=soup.find('table', id="ratings-table")
    df = pd.read_html(str(table))[0]

    int_conf=['ACC', 'SEC', 'B10', 'BSky', 'A10', 'BE']

    dic={}
    for i in int_conf:
        dic[i]=[rmv_numb2(list(df[df.iloc[:,2]==i].iloc[:,1])),list(df[df.iloc[:,2]==i].iloc[:,7]),list(df[df.iloc[:,2]==i].iloc[:,5])]
    dic["other"]=[[],[],[]]
    for j in range(len(list(df.iloc[:,2]))):
        if list(df.iloc[:,2])[j] not in int_conf:
            dic["other"][0].append(rmv_numb(df.iloc[j,1]))
            dic["other"][1].append(df.iloc[j,7])
            dic["other"][2].append(df.iloc[j,5])
        
    return dic

def exercise_2(conference):
    exercise2={ 'team':[], 'AdjO2009':[], 'diff':[]}
    for i in dic2009[conference][0]:
        for j in dic2014[conference][0]:
            if i==j:
                exercise2['team'].append(i)
                exercise2['AdjO2009'].append(dic2009[conference][2][dic2009[conference][0].index(j)])
                exercise2['diff'].append(dic2014[conference][2][dic2014[conference][0].index(j)]-dic2009[conference][2][dic2009[conference][0].index(i)])
        #Me falta implementar para que devuelva una entrada con todo lo demas :(
    return exercise2

colours=['b', 'g', 'r', 'c', 'y', 'k']
dic2014=nosemeocurrenombre("2014.html")
dic2009=nosemeocurrenombre("2009.html")

keys=list(dic2014.keys())

plt.figure()
for i in keys[:-1]:
    plt.hist(dic2014[i][1], color= colours[keys.index(i)], label=i, alpha=0.7)
plt.xlabel("AdjD in 2009")
plt.ylabel("entries")
plt.legend(loc="best", fontsize=9)

exercise2={}
plt.figure()
medians=[]
means=[]
for i in keys[:-1]:
    print(i)
    exercise2[i]=exercise_2(i)
    plt.plot(exercise2[i]['AdjO2009'], exercise2[i]['diff'], colours[keys.index(i)]+'.', label=i)
    means.append(np.mean(exercise2[i]['diff']))
    medians.append(np.median(exercise2[i]['diff']))
exercise2['other']=exercise_2('other')
means.append(np.mean(exercise2['other']['diff']))
medians.append(np.median(exercise2['other']['diff']))
plt.xlabel("AdjO in 2009")
plt.ylabel("difference in AdJO 2014-2009")
plt.legend(loc='best', fontsize=9)

raw = parser.from_file('authors.pdf') #empieza en 232,  13100
x=raw['content'][232:61006]
x=x.replace("(","<"); x=x.replace(")",">"); x=x.replace("\n", "")
x=re.sub('<[^>]+>', ',', x)
x= ''.join([i for i in x if not i.isdigit()])
x=x.replace("MMA — LIGO-P1700294-V5",""); x=x.replace("MMA — LIGO-P-V","")
x=Convert(x)
x=[i.rstrip() for i in x]; x=[i.lstrip() for i in x]
while '' in x:
    x.remove('')
x=sorted(list(set(x)), key=str.lower, reverse=True)
print("number of unique authors: ",len(x))
print("middle point of the alphabetically ordered list: ", x[int(len(x)/2)])

df2=pd.DataFrame({'Conference:':['ACC', 'SEC', 'B10', 'BSky', 'A10','BE', 'other'] , 'means':np.round(means, 3), 'medians': np.round(medians, 4)})
print(df2.to_latex(index=False))
