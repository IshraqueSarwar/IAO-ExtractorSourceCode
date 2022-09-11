import cv2
import glob
import os
import matplotlib.pyplot as plt


files = sorted(glob.glob("Lab/extracted/*.jpeg"))

last = 'x'
for file in files:
	f = file.replace('.jpeg','').split('/')
	id_ = f[-1].split('_')[0]
	if last =='x' or last !=id_:
		last = id_
	else:
		# we replace the top with white box
		img = cv2.imread(file)

		cv2.rectangle(img, (0,0), (img.shape[1],60), color = (255,255,255),thickness = -1)
		cv2.imwrite(file, img)
