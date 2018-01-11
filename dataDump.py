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
			self.img = sorted(glob.glob('./20bn-jester-v1/' + elm[0] + '/**'))
			self.imgs = []
			if len(self.img[::8]) == 5:
				cptOccur[idmove[elm[1]]] += 1
				for image in self.img[::8]:
					self.imgs.append(np.array(cv2.resize(cv2.imread(image, 0), (imgSize,imgSize))))
				self.stacked = [np.dstack(self.imgs),idmove[elm[1]]]		        
				with rlock:
					data.append(self.stacked)
				for x in [-10,-5,5,10]:#ensemble de rotation à apporter sur les images
					self.imgs = []
					for image in self.img[::8]:
						self.imgs.append(np.array(Image.fromarray(cv2.resize(cv2.imread(image, 0), (imgSize,imgSize))).rotate(x)))
					self.stacked = [np.dstack(self.imgs),idmove[elm[1]]]		        
					with rlock:
						data.append(self.stacked)
			if len(self.img[::8]) == 6:
				cptOccur[idmove[elm[1]]] += 1
				a = r.randint(0,1)
				b = 1 if not a else 0
				for image in self.img[::8][a:b]:
					self.imgs.append(np.array(cv2.resize(cv2.imread(image, 0), (imgSize,imgSize))))
				self.stacked = [np.dstack(self.imgs),idmove[elm[1]]]		        
				with rlock:
					data.append(self.stacked)
				for x in [-10,-5,5,10]:#ensemble de rotation à apporter sur les images
					self.imgs = []
					for image in self.img[::8]:
						self.imgs.append(np.array(Image.fromarray(cv2.resize(cv2.imread(image, 0), (imgSize,imgSize))).rotate(x)))
					self.stacked = [np.dstack(self.imgs),idmove[elm[1]]]		        
					with rlock:
						data.append(self.stacked)
			if len(self.img[::8]) == 7:
				for image in self.img[::8][1:-1]:
					self.imgs.append(np.array(cv2.resize(cv2.imread(image, 0), (imgSize,imgSize))))
				self.stacked = [np.dstack(self.imgs),idmove[elm[1]]]		        
				with rlock:
					data.append(self.stacked)
				for x in [-10,-5,5,10]:#ensemble de rotation à apporter sur les images
					self.imgs = []
					for image in self.img[::8]:
						self.imgs.append(np.array(Image.fromarray(cv2.resize(cv2.imread(image, 0), (imgSize,imgSize))).rotate(x)))
					self.stacked = [np.dstack(self.imgs),idmove[elm[1]]]		        
					with rlock:
						data.append(self.stacked)

def flipIndex(index):
	if index in [2,3]:
		return 2 if index == 3 else 3
	if index in [7, 8]:
		return 7 if index == 8 else 7
	if index in [15, 16]:
		return 15 if index == 16 else 15
	return index

idmove = {'Swiping Left' : 1,
		  'Swiping Right' : 2,#3
		  'Swiping Down' : 3,#2
		  'Swiping Up' : 4,
		  'Pushing Hand Away' : 5,
		  'Pulling Hand In' : 6,
		  'Sliding Two Fingers Left' : 7,#8
		  'Sliding Two Fingers Right' : 8,#7
		  'Sliding Two Fingers Down' : 9,
		  'Sliding Two Fingers Up' : 10,
		  'Pushing Two Fingers Away' : 11,
		  'Pulling Two Fingers In' : 12,
		  'Rolling Hand Forward' : 13,
		  'Rolling Hand Backward' : 14,
		  'Turning Hand Clockwise' : 15,#16
		  'Turning Hand Counterclockwise' : 16,#15
		  'Zooming In With Full Hand' : 17,
		  'Zooming Out With Full Hand' : 18,
		  'Zooming In With Two Fingers' : 19,
		  'Zooming Out With Two Fingers' : 20,
		  'Thumb Up' : 21,
		  'Thumb Down' : 22,
		  'Shaking Hand' : 23,
		  'Stop Sign' : 24,
		  'Drumming Fingers' : 25,
		  'No gesture' : 26,
		  'Doing other things' : 0
		}

nbClass = 27#len(idmove)
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
data = []

def recup(moves):
	global data

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

def dataTraitement():
	global data
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
	data = []
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

	return X_train, X_test, XClassTest, y_train, y_test, YClassTest

batch_size = 20000
for x in range(0,len(moves),batch_size):
	recup(moves[x:x+batch_size])
	print(x,len(data))

	random.shuffle(data)
	print('Chargement en RAM des images done ...')

	X_train, X_test, XClassTest, y_train, y_test, YClassTest = dataTraitement()

	cptOccur = np.array(cptOccur)
	print(cptOccur/sum(cptOccur)*100)
	X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
	XClassTest, YClassTest = np.array(XClassTest), np.array(YClassTest)
	print('Ready to dump')

	save_dir = './dataTrain/'
	if not os.path.exists(save_dir):
	    os.makedirs(save_dir)


	print("Nombres exemples d'entrainement", len(X_train))
	print("Nombres exemples de test", len(X_test))

	np.save('./dataTrain/Ytest_'+str(x), y_test)
	y_test = 0
	np.save('./dataTrain/Ytrain_'+str(x), y_train)
	y_train = 0
	np.save('./dataTrain/YtestClass_'+str(x), YClassTest)
	YClassTest = 0

	np.save('./dataTrain/XtestClass_'+str(x), XClassTest)
	XClassTest = 0
	np.save('./dataTrain/Xtest_'+str(x), X_test)
	X_test = 0
	np.save('./dataTrain/Xtrain_'+str(x), X_train)
	X_train = 0



