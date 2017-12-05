import glob
import numpy as np 
import pickle
import cv2
import random
import os


idmove = {'Swiping Left' : 0,
		  'Swiping Right' : 1,
		  'Swiping Down' : 2,
		  'Swiping Up' : 3,
		  'Pushing Hand Away' : 4,
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
		  'Doing other things' : 26
		}


nbClass = len(idmove)
cptOccur = [0 for x in range(nbClass)]
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

random.shuffle(liste)
print(len(liste))
cpt = 0
data = []
for elm in moves[:]:
	images = sorted(glob.glob('./20bn-jester-v1/' + elm[0] + '/**'))
	imgs = []
	if cpt % 1000 == 0:
		print(cpt,'/',20000)
	if cpt > 20000:
		break
	if len(images[::4]) == 10:
		cptOccur[idmove[elm[1]]] += 1
		cpt += 1
		for image in images[::4]:
			imgs.append(np.array(cv2.resize(cv2.imread(image, 0), (imgSize,imgSize))))
			"""cv2.imshow('frame', imgs[-1])
			if cv2.waitKey(100) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break"""
		data.append([np.dstack(imgs),idmove[elm[1]]])

print(len(data))

print('Chargement en RAM des images done ...')
#Traitement des images pour l'entrainement du modèle
X_train = []
y_train = []
data_train = []
for elm in data[:int(len(data)*split)]:
  classe = np.zeros(nbClass)
  classe[elm[1]] = 1
  data_train.append([np.flip(elm[0],1), classe])
  data_train.append([elm[0], classe])

print('Traitement data_train done ...')
#Traitement des images pour le test du modèle
X_test = []
y_test = []
data_test = []
for elm in data[int(len(data)*split):]:
  classe = np.zeros(nbClass)
  classe[elm[1]] = 1
  data_test.append([np.flip(elm[0],1), classe])
  data_test.append([elm[0], classe])

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


pickle.dump(X_train, open('./dataTrain/Xtrain.dump', 'wb'))
pickle.dump(y_train, open('./dataTrain/Ytrain.dump', 'wb'))
pickle.dump(X_test, open('./dataTrain/Xtest.dump', 'wb'))
pickle.dump(y_test, open('./dataTrain/Ytest.dump', 'wb'))

pickle.dump(XClassTest, open('./dataTrain/XtestClass.dump', 'wb'))
pickle.dump(YClassTest, open('./dataTrain/YtestClass.dump', 'wb'))


print("Nombres exemples d'entrainement", len(X_train))
print("Nombres exemples de test", len(X_test))
