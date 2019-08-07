#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import random as rand1
from random import randint

nb_plotted_weights = 10; #peu importe le nb de weights, on n'en mettra que 6 sur le graph

with open('../data/final_errors.txt') as f0:
    data0 = f0.read()

with open('../data/weights1.txt') as f1:
    data1 = f1.read()

with open('../data/weights2.txt') as f2:
    data2 = f2.read()

data0 = data0.split('\n')
data1 = data1.split('\n')
data2 = data2.split('\n')
data0.pop()						#delete le dernier car read() ajoute un espace
data1.pop()	
data2.pop()	
temp4 = np.array(data0,float)

nb_weights1 = data1[0].count('|') + 1	
nb_weights2 = data2[0].count('|') + 1

y1 = []	
y2 = []
randIndices1=[]	
randIndices2=[]	

for i in range(nb_plotted_weights):
	y1.append([])				
	y2.append([])						#cree juste un array vide de la bonne taille
	randIndices2.append(randint(0,nb_weights2-1))	#cree un array contenant des nb random	pour selectionner les weights aléatoirement						
	randIndices1.append(randint(0,nb_weights1-1))						
# y est une matrice 2D contenant toutes les données, 1er for necessaire pour que y[i] ne soit pas out of range
# i donne le nb de lignes différentes sur le graphe	

temp = []															
for e in data1:
	for i in range(nb_plotted_weights):
		temp = e.split('|')
		y1[i].append(temp[randIndices1[i]])
		
temp = []																	
for e in data2:
	for i in range(nb_plotted_weights):
		temp = e.split('|')
		y2[i].append(temp[randIndices2[i]])
														
		
fig0 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
#fig0.bar(np.arange(1,np.size(temp4)+1), temp4,width=0.1)	#au cas ou l'on veut faire un bar graph
fig0.plot(temp4, linewidth=0.7,c=((1,0,0.5,1)))			#rgba tous des valeurs entre 0 et 1
fig0.set_ylim(0,0.5)
fig0.set_xticks(np.arange(0,np.size(temp4),step=np.size(temp4)/10))
fig0.set_yticks(np.arange(0,0.5,step=0.1))
fig0.set_title('Final Error')
fig0.set_xlabel('Epochs')
fig0.set_ylabel('Error')	
					

fig1 = plt.subplot2grid((3, 3), (1, 0))
for e in y1:
	fig1.plot(e, linewidth=0.7)
	
fig1.set_yticks([])
fig1.set_title('Input to L1')
fig1.set_xlabel('Epochs')
fig1.set_ylabel('Weights')
	

fig2 = plt.subplot2grid((3, 3), (1, 1))
for e in y2:
	fig2.plot(e, linewidth=0.7)

fig2.set_yticks([])
fig2.set_title('L1 to Output')
fig2.set_xlabel('Epochs')
fig2.set_ylabel('Weights')

plt.subplots_adjust(wspace=0.7, hspace=0.8)

for ax in [fig0, fig1,fig2]:
	temp = []
	#for e in plt.axes(ax).get_yticks():
	#	temp.append(round(e))		#attempt to round the values of the ticks to 2 chiffres significatifs
	#ax.set_yticklabels(temp)
	
plt.savefig('../data/plots.png')
plt.show()
