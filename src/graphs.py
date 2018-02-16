#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

with open('../data/final_errors.txt') as f:
    data1 = f.read()

data1 = data1.split('\n')
data1.pop()			#delete le dernier car read() ajoute un espace

temp4 = np.array(data1)
temp5 = np.arange(temp4.size)

for i in temp4:			
	i = round(float(i), 3)
	
fig0 = plt.subplot2grid((2, 2), (1, 1))
fig0.scatter(temp5, temp4, s=0.7, marker='x',c=((1,0,0.5,1)))			#rgba tous des valeurs entre 0 et 1
fig0.set_title('Final Error')
fig0.set_xlabel('Epochs')
fig0.set_ylabel('Error')


with open('../data/weights3.txt') as f:
    data3 = f.read()

data3 = data3.split('\n')
data3.pop()	

nb_weights = data3[0].count('|') + 1	

y3 = []	
for i in range(nb_weights):
	y3.append([])													# y est une matrice 2D contenant toutes les données, 1er for necessaire pour que y[i] ne soit pas out of range
temp = []															# i donne le nb de lignes différentes sur le graphe
for e in data3:
	for i in range(nb_weights):
		temp = e.split('|')
		y3[i].append(temp[i])
	
with open('../data/weights2.txt') as f:
    data2 = f.read()

data2 = data2.split('\n')
data2.pop()	

temp = data2[0].count('|') + 1
if(temp< 15):
	nb_weights = temp    			# il y a trop de connections --> on selectionne 15
else:
	nb_weights = 15

y2 = []	
for i in range(nb_weights):
	y2.append([])													
temp = []															
for e in data2:
	for i in range(nb_weights):
		temp = e.split('|')
		y2[i].append(temp[i])

with open('../data/weights1.txt') as f:
    data1 = f.read()

data1 = data1.split('\n')
data1.pop()	

nb_weights = data1[0].count('|') + 1	

y1 = []	
for i in range(nb_weights):
	y1.append([])													
temp = []															
for e in data1:
	for i in range(nb_weights):
		temp = e.split('|')
		y1[i].append(temp[i])

fig1 = plt.subplot2grid((2, 2), (0, 0))
for e in y1:
	fig1.plot(e, linewidth=0.7)
fig1.set_title('Weights of Input to L1')
fig1.set_xlabel('Epochs')
fig1.set_ylabel('Weights')
	
fig2 = plt.subplot2grid((2, 2), (0, 1))
for e in y2:
	fig2.plot(e, linewidth=0.7)
fig2.set_title('Weights of L1 to L2')
fig2.set_xlabel('Epochs')
fig2.set_ylabel('Weights')
	
fig3 = plt.subplot2grid((2, 2), (1, 0))
for e in y3:
	fig3.plot(e, linewidth=0.7)
fig3.set_title('Weights of L2 to Output')
fig3.set_xlabel('Epochs')
fig3.set_ylabel('Weights')

plt.axes(fig0).get_yaxis().set_ticks([])
plt.axes(fig1).get_yaxis().set_ticks([])
plt.axes(fig2).get_yaxis().set_ticks([])
plt.axes(fig3).get_yaxis().set_ticks([])
plt.subplots_adjust(wspace=0.5, hspace=0.6)
plt.savefig('plots.png')
plt.show()
