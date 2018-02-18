#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

with open('../data/final_errors.txt') as f:
    data1 = f.read()

data1 = data1.split('\n')
data1.pop()			#delete le dernier car read() ajoute un espace

temp4 = np.array(data1)

with open('../data/weights2.txt') as f:
    data2 = f.read()

data2 = data2.split('\n')
data2.pop()	

nb_weights = data2[0].count('|') + 1
if(nb_weights > 7):
	nb_weights = 7    			# il y a trop de connections --> on selectionne 10

y2 = []	
for i in range(nb_weights):
	y2.append([])													
														# y est une matrice 2D contenant toutes les données, 1er for necessaire pour que y[i] ne soit pas out of range
temp = []															# i donne le nb de lignes différentes sur le graphe											
for e in data2:
	for i in range(nb_weights):
		temp = e.split('|')
		y2[i].append(temp[i])

with open('../data/weights1.txt') as f:
    data1 = f.read()

data1 = data1.split('\n')
data1.pop()	

nb_weights = data1[0].count('|') + 1	
if(nb_weights > 7):
	nb_weights = 7 

y1 = []	
for i in range(nb_weights):
	y1.append([])													
temp = []															
for e in data1:
	for i in range(nb_weights):
		temp = e.split('|')
		y1[i].append(temp[i])
		
fig0 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
fig0.plot(temp4, linewidth=0.7,c=((1,0,0.5,1)))			#rgba tous des valeurs entre 0 et 1
fig0.set_title('Final Error')
fig0.set_xlabel('Epochs')
fig0.set_ylabel('Error')						

fig1 = plt.subplot2grid((2, 2), (1, 0))
for e in y1:
	fig1.plot(e, linewidth=0.7)
fig1.set_title('Input to L1')
fig1.set_xlabel('Epochs')
fig1.set_ylabel('Weights')
	
fig2 = plt.subplot2grid((2, 2), (1, 1))
for e in y2:
	fig2.plot(e, linewidth=0.7)
fig2.set_title('L1 to Output')
fig2.set_xlabel('Epochs')
fig2.set_ylabel('Weights')

plt.subplots_adjust(wspace=0.7, hspace=0.5)

#fig0.set_ylim(-1.1,1.1)
#fig0.set_yticks((-1.0,0.0,1.0))
#fig0.set_yticklabels(('-1.0','0.0','1.0'))

#fig0.set_yticks(np.arange(min(data1), max(data1)+1, 1.0))

for ax in [fig0, fig1,fig2]:
	temp = []
	#for e in plt.axes(ax).get_yticklabels():
	#	temp.append(str(round(int(e), 2)))
	#ax.set_yticklabels(temp)
	ax.locator_params(axis='y', nbins=5)
	
plt.savefig('plots.png')
plt.show()
