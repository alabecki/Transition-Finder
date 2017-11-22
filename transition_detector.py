import numpy.core._methods
import numpy.lib.format
import numpy as np
#from PIL import Image, ImageTk
import cv2
from appJar import gui
#import threading
#from threading import Thread
import time
import sys
import math


def chromatize(pixel):
	for f in range(NUMBER_OF_FRAMES):
    	b,g,r = cv2.split(bgr_img)

    	r = r/r+g+b
    	b = 
    	#rgb_img = cv2.merge([r,g,b])

def chrom_bin(pixel):
	return

class CHistogram(object):
	def __init__(self, column, frame)
		global WIDTH, HEIGHT
		self.x = log(WIDTH, 2)
		self.y = log(HEIGHT, 2)
		#np.zeros( (x,y))
		self.histogram = [[0] * x for i in range(y)]
		self.column = 0
		self.frame


	def reduce_histogram(self):
		sum = 0
		for i in range(self.x):
			for j in range(self.y):
				sum += self.histogram[i][j]
		for i in range(self.x):
			for j in range(self.y):
				self.histogram[i][j] = self.histogram[i][j]/sum

	def column_frame_histogram(self, column, frame):
		for pixel in column:
			self.histogram[pixel[0]][pixel[1]]++


def histo_difference(t1, t2):
	value = 0
	for i in range(t1.width):
		for j in range(t.height):
			value += min(t1[i][j], t2[i][j])
	return value

def colour_transformation():
	global WIDTH, HEIGHT
	for r in range HEIGHT
		for c 

def create_chistograms():
	histograms = []
	for c in range(WIDTH):
		for t in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
			hist = CHistogram(c, t)



def create_histo_STI(histograms):
	x = height    										#columns
	y = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))			#frames
	for c in range(x):
		for t in range(y): 
		hist = Hist(c, t)






def open_video(btn):
        global file     
        frame = np.zeros(shape = (width, height))
        file = app.openBox(title= "Choose a Video", dirName=None, fileTypes=None, asFile=True, parent=None)
        cap = cv2.VideoCapture(file.name)
        ret, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        app.reloadImageData("Video", img, fmt="mpg")
        cap.release()
        return





def build_column_STI():
    global NUMBER_OF_FRAMES
    STIlist = []
    for c in range(width):
        frame = np.zeros(shape = (NUMBER_OF_FRAMES, height)) 
        for t in range(NUMBER_OF_FRAMES):
            cap.set(1, t)
            ret, frame = cap.read()
            if ret: 

            frame[ :, t] =  temp[: , c] 
        STIlist = STIlist.append(frame)  




