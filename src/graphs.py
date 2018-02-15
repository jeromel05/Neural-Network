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

with open('../data/weights3.txt') as f:
    data1 = f.read()

data1 = data1.split('\n')
data1.pop()		

y1 = []
y2 = []
for e in data1:
	temp = e.split('|')
	y1.append(temp[0])
	y2.append(temp[1])


fig3 = plt.subplot2grid((2, 2), (0, 0))
fig3.plot(y1)
fig3.plot(y2)
fig3.set_title('Weights of third connections')

plt.show()
