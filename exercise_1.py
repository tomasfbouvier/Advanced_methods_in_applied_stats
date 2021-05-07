# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:09:59 2021

@author: To+
"""

import re
import numpy as np

rx_dict={ "Data set":re.compile(r'Data set' ) }   

def parse_file(filepath):
    data={}
    with open(filepath, 'r') as file_object:
        next(file_object)
        line=file_object.readline()         
        while line:
            for key, rx in rx_dict.items():
                match= rx.search(line)
                if match:
                    line = line.replace('\n','')
                    name=line
                    data[name]=[[],[]]
                    line = file_object.readline()  
            b=line.strip().split('\t')
            if(b!=['']):
                b=[float(i) for i in b]
                data[name][0].append(b[0])
                data[name][1].append(b[1])
            line = file_object.readline()
    return data

data=parse_file("FranksNumbers.txt")

def chi2(x,y, )          