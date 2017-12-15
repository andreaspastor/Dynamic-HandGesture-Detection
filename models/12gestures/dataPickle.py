import glob
import numpy as np 
import cv2
import random
import os
import sys
from PIL import Image
from threading import Thread, RLock
from time import time

rlock = RLock()

class OpenImage(Thread):
	""" Thread to open images. """
	def __init__(self, listA):
	    global data, cptOccur, idmove
	    Thread.__init__(self)
	    self.listA = listA
	    self.img, self.value, self.imgs = None, None, None

	def run(self):
		""" Code to execute to open. """
		for elm in self.listA:
			if idmove[elm[1]] in [0,1,2,3,4,5,6,7,8,9,10,11,12]:
				self.img = sorted(glob.glob('./20bn-jester-v1/' + elm[0] + '/**'))
				self.imgs = []
				if len(self.img[::8]) == 5:
					cptOccur[idmove[elm[1]]] += 1
					for image in self.img[::8]:
						self.imgs.append(np.array(cv2.resize(cv2.imread(image, 0), (imgSize,imgSize))))
					self.stacked = [np.dstack(self.imgs),idmove[elm[1]]]		        
					with rlock:
						data.append(self.stacked)
					for x in [-5,5]:
						self.imgs = []
						for image in self.img[::8]:
							self.imgs.append(np.array(Image.fromarray(cv2.resize(cv2.imread(image, 0), (imgSize,imgSize))).rotate(x)))
						self.stacked = [np.dstack(self.imgs),idmove[elm[1]]]		        
						with rlock:
							data.append(self.stacked)

idmove = {'Swiping Left' : 0,
		  'Swiping Right' : 1,
		  'Swiping Down' : 2,
		  'Swiping Up' : 3,
		  'Pushing Hand Away' : 26,
		  'Pulling Hand In' : 5,
		  'Sliding Two Fingers Left' : 6,
		  'Sliding Two Fingers Right' : 7,
		  'Sliding Two Fingers Down' : 8,
		  'Sliding Two Fingers Up' : 9,
		  'Pushing Two Fingers Away' : 10,
		  'Pulling Two Fingers In' : 11,
		  'Rolling Hand Forward' : 12,
		  'Rolling Hand Backward' : 13,
		  'Turning Hand Clockwise' : 14,
		  'Turning Hand Counterclockwise' : 15,
		  'Zooming In With Full Hand' : 16,
		  'Zooming Out With Full Hand' : 17,
		  'Zooming In With Two Fingers' : 18,
		  'Zooming Out With Two Fingers' : 19,
		  'Thumb Up' : 20,
		  'Thumb Down' : 21,
		  'Shaking Hand' : 22,
		  'Stop Sign' : 23,
		  'Drumming Fingers' : 24,
		  'No gesture' : 25,
		  'Doing other things' : 4
		}

nbClass = 5#len(idmove)
cptOccur = [0 for x in range(len(idmove))]
split = 0.9
imgSize = 64
liste = glob.glob("./20bn-jester-v1/**")

moves = []
file = open('jester-v1-train.csv', 'r')
ligne = file.readline()
while ligne != "":
	ligne = ligne.split(';')
	print(ligne)
	moves.append([ligne[0], ligne[1][:-1]])
	ligne = file.readline()

file.close()

random.shuffle(moves)
print(len(moves))
#moves = moves[:100000]
print(len(moves))
data = []

#Chargement en RAM des images trouvees
# Threads Creation
t1 = time()
threads = []

nbThread = 20
size = int(len(moves)/nbThread)
for x in range(nbThread):
    threads.append(OpenImage(moves[x*size:(x+1)*size]))

# Lancement des threads
for thread in threads:
    thread.start()


# Attend que les threads se terminent
for thread in threads:
    thread.join()


print(len(data))

random.shuffle(data)
print('Chargement en RAM des images done ...')
#Traitement des images pour l'entrainement du modele
X_train = []
y_train = []
data_train = []
for elm in data[:int(len(data)*split)]:
	classe = np.zeros(nbClass)
	classe[elm[1]] = 1
	data_train.append([elm[0], classe])
	if elm[1] == 0:
		classe = np.zeros(nbClass)
		classe[1] = 1
	elif elm[1] == 1:
		classe = np.zeros(nbClass)
		classe[0] = 1
	data_train.append([np.flip(elm[0],1), classe])
  

print('Traitement data_train done ...')
#Traitement des images pour le test du modele
X_test = []
y_test = []
data_test = []
for elm in data[int(len(data)*split):]:
	classe = np.zeros(nbClass)
	classe[elm[1]] = 1
	data_test.append([elm[0], classe])
	if elm[1] == 0:
		classe = np.zeros(nbClass)
		classe[1] = 1
	elif elm[1] == 1:
		classe = np.zeros(nbClass)
		classe[0] = 1
	data_test.append([np.flip(elm[0],1), classe])

print('Traitement data_test done ...')
data = 0
random.shuffle(data_test)
random.shuffle(data_train)

XClassTest = [[] for x in range(nbClass)]
YClassTest = [[] for y in range(nbClass)]
for elm in data_test:
  x = np.argmax(elm[1])
  YClassTest[x].append(elm[1])
  XClassTest[x].append(elm[0])


for elm in data_train:
  X_train.append(elm[0])
  y_train.append(elm[1])
data_train = 0

for elm in data_test:
  X_test.append(elm[0])
  y_test.append(elm[1])
data_test = 0

cptOccur = np.array(cptOccur)
print(cptOccur/sum(cptOccur)*100)
X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
XClassTest, YClassTest = np.array(XClassTest), np.array(YClassTest)
print('Ready to dump')

save_dir = './dataTrain/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


np.save('./dataTrain/Xtrain', X_train)
np.save('./dataTrain/Ytrain', y_train)
np.save('./dataTrain/Xtest', X_test)
np.save('./dataTrain/Ytest', y_test)

np.save('./dataTrain/XtestClass', XClassTest)
np.save('./dataTrain/YtestClass', YClassTest)


print("Nombres exemples d'entrainement", len(X_train))
print("Nombres exemples de test", len(X_test))
