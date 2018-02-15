#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def affiche_Matrice(arr):
	for line in arr:
			print(line)
	return arr.dtype
	
def affiche_Ligne(l):
	for i in l:
		print(i, 4)
		
		
with open('../data/final_errors.txt') as f:
    data1 = f.read()

data1 = data1.split('\n')
data1.pop()			#delete le dernier car read() ajoute un espace

temp4 = np.array(data1)
temp5 = np.arange(temp4.size)

for i in temp4:			
	i = round(float(i), 3)
	
temp2 = plt.scatter(temp5, temp4, s=1, marker='D',c=((1,0,0.5,1)))			#rgba tous des valeurs entre 0 et 1

plt.savefig('plots.png')
plt.show()
